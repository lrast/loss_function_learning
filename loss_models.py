# models to capture a surrogate for classification loss.

import torch
import weakref

from transformers import ViTConfig
import pytorch_lightning as pl
from transformers.models.vit.modeling_vit import ViTEncoder

from torch.utils.data import IterableDataset

from dataclasses import dataclass
from typing import Optional
from transformers.utils import ModelOutput

from transformers import Trainer, TrainingArguments


@dataclass
class GradientModelOutput(ModelOutput):
    """
    Base class for outputs of the gradient models
    """
    predictions: torch.FloatTensor = None 
    loss: Optional[torch.FloatTensor] = None
    labels: Optional[torch.LongTensor] = None


class EmbeddingToGradient(torch.nn.Module):
    """EmbeddingToGradient: Transformer model that directly maps
    image embeddings to loss function gradients
    """
    def __init__(self, num_hidden_layers=2):
        super().__init__()
        config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224",
            num_hidden_layers=num_hidden_layers,
            attn_implementation="eager",  # change this for performance
            id2label=None,
            label2id=None
        )

        self.main_input_name = "embeddings"

        self.local_gradient_approximation = ViTEncoder(config)
        self.loss = torch.nn.MSELoss()

    def forward(self, embeddings, targets=None, labels=None):
        raw_outputs = self.local_gradient_approximation(embeddings)

        loss = None
        if targets is not None:
            loss = self.loss(raw_outputs.last_hidden_state, targets)

        return GradientModelOutput(
                                   predictions=raw_outputs.last_hidden_state,
                                   loss=loss,
                                   labels=labels
                                  )




class BatchRecorder:
    """BatchRecorder: records input, gradient pairs for the module specified """
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
                                             **kwargs),
        }
        if val_data is not None:
            self.datasets["val"] = ActivityGradientDataset(model, recorder, val_data,
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
                 device=None
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

    def __iter__(self):
        if self.shuffle:
            inds = torch.randperm(len(self.raw_inputs))
        else:
            inds = torch.arange(len(self.raw_inputs))

        batches = [inds[i: i+self.batch_size].tolist() 
                   for i in range(0, len(self.raw_inputs), self.batch_size)
                   ]

        for batch_inds in batches:
            # Forward and backward passes through the network
            images, labels = self.raw_inputs[batch_inds]
            images = self.model.preprocess(images).to(self.device)
            outs = self.model.forward(images, labels=labels.to(self.device))
            outs.loss.backward()

            yield (*self.recorder.batch, labels)


def make_gradient_trainer(classifier_model, gradient_model, train, val,
                          epochs=10, step_size=1E-1, **kwargs):

    default_ds_args = {
                        'batch_size': 8,
                        'shuffle': True,
                        'rand_seed': 42,
                        'device': 'mps'
                        }
    ds_args = {k: kwargs.pop(k, v) for k, v in default_ds_args.items()}
    activity_ds = ActivityGradientDataDict(classifier_model, classifier_model.embedding.vit,
                                           train, val,
                                           **default_ds_args)

    default_train_args = {
                          'learning_rate': 5E-5,
                          'num_train_epochs': epochs,
                          'max_steps': epochs * len(train) // ds_args['batch_size'],
                          'per_device_train_batch_size': 1,

                          'weight_decay': 0.01,

                          'lr_scheduler_type': 'constant',
                          'warmup_ratio': 0.0,

                          'logging_steps': 20,
                          'logging_strategy': "steps",
                          'output_dir': 'gradient_overfit_large',

                          'include_for_metrics': ['inputs', 'loss'],
                          'eval_strategy': 'epoch',
                          'save_total_limit': 1,
                        }
    train_args = {**default_train_args, **kwargs}
    training_args = TrainingArguments(train_args)

    def batch_collator(data):
        return {'embeddings': data[0][0],
                'targets': data[0][1],
                'labels': data[0][2]
                }

    def classification_output(classifier_model, embeddings): 
        x = classifier_model.classifier.vit.encoder(embeddings).last_hidden_state
        x = classifier_model.classifier.vit.layernorm(x)[:, 0, :]
        logits = classifier_model.classifier.classifier(x)

        sm_outputs = torch.nn.functional.softmax(logits, dim=1)
        return sm_outputs

    def compute_metrics(preds):
        """Compute change in loss after a single gradient step"""
        gradient = torch.as_tensor(preds.predictions[0], device=ds_args['device'])
        embedding = torch.as_tensor(preds.inputs, device=ds_args['device'])
        labels = torch.as_tensor(preds.label_ids, device=ds_args['device'])

        perturbed_embedding = embedding - step_size * gradient

        baseline_probs = classification_output(classifier_model, embedding)
        perturbed_probs = classification_output(classifier_model, perturbed_embedding)

        label_probs_baseline = baseline_probs[range(len(labels)), labels.tolist()]
        label_probs_perturbed = perturbed_probs[range(len(labels)), labels.tolist()]

        base_class = baseline_probs.argmax(1)
        perturbed_class = perturbed_probs.argmax(1)

        frac_improved = (label_probs_perturbed > label_probs_baseline).sum() / ds_args['batch_size']
        dCorrect = ((perturbed_class == labels).sum() - (base_class == labels).sum()) / ds_args['batch_size']

        return {'improved_probability': frac_improved,
                'improved_classification': dCorrect}

    trainer = Trainer(gradient_model,
                      args=training_args,
                      train_dataset=activity_ds['train'],
                      eval_dataset=activity_ds['val'],
                      data_collator=batch_collator,
                      compute_metrics=compute_metrics
                      )
    trainer.can_return_loss = True

    return trainer


