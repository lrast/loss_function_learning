# models to capture a surrogate for classification loss.

import torch
import pytorch_lightning as pl

from torch import nn

from dataclasses import dataclass
from typing import Optional

from transformers import ViTConfig
from transformers.utils import ModelOutput
from transformers.models.vit.modeling_vit import ViTEncoder


@dataclass
class RegressionModelOutput(ModelOutput):
    """
    Base class for outputs of these regression models
    """
    predictions: torch.FloatTensor = None 
    loss: Optional[torch.FloatTensor] = None


class EmbeddingToGradient(pl.LightningModule):
    """EmbeddingToGradient: Transformer model that directly maps
    image embeddings to loss function gradients
    """
    def __init__(self, num_hidden_layers=6, classifier_model=None, step_size=100,
                 only_first=False
                 ):
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
            # mimic the effects of step size in the loss function
            loss = self.loss(self.step_size*predictions, self.step_size*targets)

        return RegressionModelOutput(
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

            baseline_probs = torch.nn.functional.softmax(
                                    classification_output(self.classifier_model,
                                                          embedding),
                                    dim=1)
            perturbed_probs = torch.nn.functional.softmax(
                                    classification_output(self.classifier_model,
                                                          perturbed_embedding),
                                    dim=1)

            label_probs_baseline = baseline_probs[range(len(labels)), labels.tolist()]
            label_probs_perturbed = perturbed_probs[range(len(labels)), labels.tolist()]

            base_class = baseline_probs.argmax(1)
            perturbed_class = perturbed_probs.argmax(1)

            frac_improved = ((label_probs_perturbed > label_probs_baseline).sum()
                             ) / embedding.shape[0]
            dCorrect = ((perturbed_class == labels).sum() - (base_class == labels).sum()
                        ) / embedding.shape[0]

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


class EmbeddingPropagation(pl.LightningModule):
    """EmbeddingPropagation: evolves the Embedding to simulate a gradient step

    Only first: learns only the propagation of the first token in the hidden
    representaion
    """
    def __init__(self, num_hidden_layers=6, classifier_model=None,
                 step_size=100, only_first=False, lr=5e-5,
                 FC_model=False
                 ):
        super().__init__()

        if FC_model:
            print('here')
            self.propagation_model = nn.Sequential(*(num_hidden_layers*[
                                                      nn.ReLU(),
                                                      nn.Linear(768, 768)
                                                    ])
                                                   )

        else:
            config = ViTConfig.from_pretrained(
                "google/vit-base-patch16-224",
                num_hidden_layers=num_hidden_layers,
                attn_implementation="eager",  # change this for performance
                id2label=None,
                label2id=None
            )

            self.propagation_model = ViTEncoder(config)

        if only_first:
            self.loss = lambda inputs, targets: torch.nn.functional.mse_loss(
                                            inputs[:, 0, :], targets[:, 0, :])
        else:
            self.loss = torch.nn.MSELoss()

        self.main_input_name = "embeddings"
        self.classifier_model = classifier_model
        self.step_size = step_size
        self.only_first = only_first
        self.lr = lr
        self.FC_model = FC_model

    def forward(self, embeddings, targets=None):
        if self.FC_model:
            predictions = embeddings.clone()
            predictions[:, 0, :] = self.propagation_model(embeddings[:, 0, :])

        else:
            predictions = self.propagation_model(embeddings).last_hidden_state

        if self.only_first:
            predictions[:, 1:, :] = embeddings[:, 1:, :]

        loss = None
        if targets is not None:
            loss = self.loss(predictions, targets)

        return RegressionModelOutput(
                                   predictions=predictions,
                                   loss=loss
                                  )

    def training_step(self, batch, batchid=None):
        embedding, gradient = map(lambda x: x.squeeze(), batch)

        # Now the target is the embedding after evolution along the gradient
        perturbed_embedding = embedding - self.step_size * gradient

        loss = self.forward(embedding, targets=perturbed_embedding).loss
        self.log("train/loss", loss.item(), on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batchid=None):
        """Validation metrics that track the impact of gradient of classification
        performance.
        """
        embedding, gradient, labels = map(lambda x: x.squeeze(), batch)
        perturbed_embedding_gt = embedding - self.step_size * gradient

        outputs = self.forward(embedding, targets=perturbed_embedding_gt)

        perturbed_embedding_pred = outputs.predictions
        loss = outputs.loss

        self.log("eval/loss", loss, prog_bar=True)

        if self.classifier_model is not None:
            baseline_probs = torch.nn.functional.softmax(
                                        classification_output(self.classifier_model,
                                                              embedding),
                                        dim=1)
            perturbed_probs = torch.nn.functional.softmax(
                                        classification_output(self.classifier_model,
                                                              perturbed_embedding_pred),
                                        dim=1)

            label_probs_baseline = baseline_probs[range(len(labels)), labels.tolist()]
            label_probs_perturbed = perturbed_probs[range(len(labels)), labels.tolist()]

            base_class = baseline_probs.argmax(1)
            perturbed_class = perturbed_probs.argmax(1)

            frac_improved = ((label_probs_perturbed > label_probs_baseline).sum()
                             ) / embedding.shape[0]
            dCorrect = ((perturbed_class == labels).sum() - (base_class == labels).sum()
                        ) / embedding.shape[0]

            self.log("eval/improved_probability", frac_improved)
            self.log("eval/improved_classification", dCorrect)

    def configure_optimizers(self):
        """Hugging Face-style AdamW with parameter groups excluding bias and
        LayerNorm from weight decay"""
        learning_rate = self.lr
        weight_decay = 0.0
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

    return logits
