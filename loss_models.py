# models to capture a surrogate for classification loss.

import torch
import weakref
import pytorch_lightning as pl

from transformers import ViTConfig
from transformers.utils import ModelOutput
from transformers.models.vit.modeling_vit import ViTEncoder

from torch.utils.data import IterableDataset,  DataLoader

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from dataclasses import dataclass
from typing import Optional


@dataclass
class GradientModelOutput(ModelOutput):
    """
    Base class for outputs of the gradient models
    """
    predictions: torch.FloatTensor = None 
    loss: Optional[torch.FloatTensor] = None


class EmbeddingToGradient(pl.LightningModule):
    """EmbeddingToGradient: Transformer model that directly maps
    image embeddings to loss function gradients
    """
    def __init__(self, num_hidden_layers=2, classifier_model=None, step_size=1E-1):
        super().__init__()
        config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224",
            num_hidden_layers=num_hidden_layers,
            attn_implementation="eager",  # change this for performance
            id2label=None,
            label2id=None
        )

        self.local_gradient_approximation = ViTEncoder(config)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(768, 768),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(768, 768)
                                           )

        self.loss = torch.nn.L1Loss()

        self.main_input_name = "embeddings"
        self.classifier_model = classifier_model
        self.step_size = step_size

    def forward(self, embeddings, targets=None):
        raw_outputs = self.local_gradient_approximation(embeddings)
        predictions = self.decoder(raw_outputs.last_hidden_state)

        loss = None
        if targets is not None:
            loss = self.loss(predictions / torch.norm(predictions, dim=(1, 2))[:, None, None],
                             targets / torch.norm(targets, dim=(1, 2))[:, None, None])

        return GradientModelOutput(
                                   predictions=predictions,
                                   loss=loss
                                  )

    def training_step(self, batch, batchid=None):
        embedding, gradients = map(lambda x: x.squeeze(), batch)

        loss = self.forward(embedding, gradients).loss
        self.log("train/loss", loss.item(), on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batchid=None):
        """Validation metrics that track the impact of gradient of classification
        performance.
        """
        embedding, gradient_gt, labels = map(lambda x: x.squeeze(), batch)
        outputs = self.forward(embedding, targets=gradient_gt)

        gradient_pred = outputs.predictions
        loss = outputs.loss

        self.log("eval/loss", loss, prog_bar=True)

        if self.classifier_model is not None:
            perturbed_embedding = embedding - self.step_size * gradient_pred

            baseline_probs = classification_output(self.classifier_model, embedding)
            perturbed_probs = classification_output(self.classifier_model, perturbed_embedding)

            label_probs_baseline = baseline_probs[range(len(labels)), labels.tolist()]
            label_probs_perturbed = perturbed_probs[range(len(labels)), labels.tolist()]

            base_class = baseline_probs.argmax(1)
            perturbed_class = perturbed_probs.argmax(1)

            frac_improved = (label_probs_perturbed > label_probs_baseline).sum() / embedding.shape[0]
            dCorrect = ((perturbed_class == labels).sum() - (base_class == labels).sum()) / embedding.shape[0]

            self.log("eval/improved_probability", frac_improved)
            self.log("eval/improved_classification", dCorrect)

    def configure_optimizers(self):
        """Hugging Face-style AdamW with parameter groups excluding bias and
        LayerNorm from weight decay"""
        learning_rate = 5e-5
        weight_decay = 0.01
        betas = (0.9, 0.999)
        eps = 1e-8

        decay_parameters = []
        no_decay_parameters = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or name.endswith("LayerNorm.weight"):
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)

        param_groups = [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
        )

        return optimizer


def classification_output(classifier_model: torch.nn.Module, embeddings: torch.Tensor
                          ) -> torch.Tensor: 
    x = classifier_model.classifier.vit.encoder(embeddings).last_hidden_state
    x = classifier_model.classifier.vit.layernorm(x)[:, 0, :]
    logits = classifier_model.classifier.classifier(x)

    sm_outputs = torch.nn.functional.softmax(logits, dim=1)
    return sm_outputs


class BatchRecorder:
    """BatchRecorder: records input, gradient pairs for the module specified"""
    def __init__(self, module):
        self.batch = None
        self.hooks = module.register_forward_hook(self.hook_fn)
        weakref.finalize(self, self.cleanup)

    def hook_fn(self, module, input, output):
        """Forward hook: attaches a gradient hook to the module's output."""
        out = output.last_hidden_state
        # Capture the forward output (detach so it’s not tied to graph)
        out_detached = out.detach().clone()

        def grad_hook(grad, out_copy=out_detached):
            # grad will be populated during backward
            self.batch = (out_copy, grad.detach().clone())

        out.register_hook(grad_hook)

    def cleanup(self):
        # Remove hooks
        self.hooks.remove()

    def __del__(self):
        self.cleanup()


class ActivityGradientDataDict:
    """Activity-gradient datasets: training and validation
    """
    def __init__(self, model, module, train_data, val_data=None, **kwargs):
        recorder = BatchRecorder(module)
        self.datasets = {
            "train": ActivityGradientDataset(model, recorder, train_data,
                                             return_labels=False,
                                             **kwargs),
        }
        if val_data is not None:
            self.datasets["val"] = ActivityGradientDataset(model, recorder,
                                                           val_data,
                                                           return_labels=True,
                                                           **kwargs)

    def __getitem__(self, key):
        return self.datasets[key]


class ActivityGradientDataset(IterableDataset):
    """Dateset of activity and loss function gradients for a given module
    model: parent model containing module of interest
    recorder: an activity / gradient recorder object
    raw_inputs: inputs to parent model to iterate over
    """
    def __init__(self, model, recorder, raw_inputs,
                 batch_size=8, shuffle=True, rand_seed=42,
                 device=None, return_labels=False
                 ):
        super(ActivityGradientDataset).__init__()

        self.model = model.to(device)
        self.raw_inputs = raw_inputs

        # Initialize model hooks
        self.recorder = recorder

        # Initialize random state
        self.rng = torch.Generator()
        self.rng.manual_seed(43)

        # Batch parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.return_labels = return_labels

    def __iter__(self):
        if self.shuffle:
            inds = torch.randperm(len(self.raw_inputs))
        else:
            inds = torch.arange(len(self.raw_inputs))

        batches = [inds[i: i+self.batch_size].tolist() 
                   for i in range(0, len(self.raw_inputs), self.batch_size)
                   ]

        with torch.enable_grad():
            for batch_inds in batches:
                # Forward and backward passes through the network
                images, labels = self.raw_inputs[batch_inds]
                images = self.model.preprocess(images).to(self.device)
                images.requires_grad = True
                outs = self.model.forward(images, labels=labels.to(self.device))
                outs.loss.backward()

                if self.return_labels:
                    yield (*self.recorder.batch, labels)
                else:
                    yield self.recorder.batch


def make_gradient_trainer(classifier_model, gradient_model,
                          images_train, images_val, 
                          project_name='TTA_loss', run_name='debug',
                          checkpoint_directory='debug',
                          epochs=10, step_size=1E-1, **kwargs):
    """Make a pytorch lightgning trainer for the gradient model"""

    default_ds_args = {
                        'batch_size': 8,
                        'shuffle': True,
                        'rand_seed': 42,
                        'device': 'mps'
                        }
    ds_args = {k: kwargs.pop(k, v) for k, v in default_ds_args.items()}
    activity_ds = ActivityGradientDataDict(
        classifier_model, classifier_model.embedding.vit,
        images_train,  images_val,
        **ds_args
    )

    # DataLoaders
    train_loader = DataLoader(activity_ds['train'], batch_size=1)
    val_loader = DataLoader(activity_ds['val'], batch_size=1)

    # Set up PyTorch Lightning Trainer
    default_trainer_args = {
        'max_epochs': epochs,
        'accelerator': 'auto',
        'log_every_n_steps': 20,
        'enable_progress_bar': True,
    }
    trainer_args = {**default_trainer_args, **kwargs}
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_directory,
        save_top_k=1,
        monitor='eval/loss',
        mode='min',
        save_last=True,
        every_n_epochs=1
    )

    wandb_logger = WandbLogger(name=run_name, project=project_name)

    pl_trainer = pl.Trainer(
        **trainer_args,
        logger=[wandb_logger],
        callbacks=[checkpoint]
    )

    return pl_trainer, gradient_model, train_loader, val_loader
