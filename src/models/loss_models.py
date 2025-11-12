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

    starting_point: Optional[torch.FloatTensor] = None
    labels: Optional[torch.IntTensor] = None


class BaseEmbeddingRegressor(pl.LightningModule):
    """
    Base class for embedding regression models
    """
    def __init__(self, classifier_model=None, step_size=100,
                 num_transformer_layers=6, num_fc_layers=2,
                 lr=5e-5,
                 ):
        super().__init__()
        # global settings 
        self.main_input_name = "embeddings"
        self.classifier_model = classifier_model
        self.step_size = step_size

        # Model initialization
        if num_transformer_layers > 0:
            config = ViTConfig.from_pretrained(
                "google/vit-base-patch16-224",
                num_hidden_layers=num_transformer_layers,
                attn_implementation="eager",  # change this for performance
                id2label=None,
                label2id=None
            )

            self.transformer_layers = ViTEncoder(config)
        else:
            self.transformer_layers = nn.Identity()

        if num_fc_layers > 0: 
            self.fc_layers = nn.Sequential(*(
                                            [nn.Linear(768, 768)] +
                                            [
                                              nn.ReLU(),
                                              nn.Linear(768, 768)
                                            ] * (num_fc_layers-1)
                                           ))
        else:
            self.fc_layers = nn.Identity()

        self.lr = lr
        self.loss = nn.MSELoss()

    def forward(self, embeddings, targets=None) -> RegressionModelOutput:
        """ Applies the models in order, compute loss """
        predictions = self.fc_layers(self.transformer_layers(embeddings).last_hidden_state)

        loss = None
        if targets is not None:
            # mimic the effects of step size in the loss function
            loss = self.loss(predictions, targets)

        return RegressionModelOutput(
                           predictions=predictions,
                           loss=loss
                          )

    def next_embedding(self, batch) -> RegressionModelOutput:
        """ predicts the embedding values after a gradient step """
        pass

    def training_step(self, batch, batchid=None):
        pass

    def validation_step(self, batch, batchid=None):
        """Validation metrics that track the impact of gradient of classification
        performance.
        """
        predictions = self.next_embedding(batch)

        loss = predictions.loss
        start_embedding = predictions.starting_point
        perturbed_embedding = predictions.predictions
        labels = predictions.labels

        self.log("eval/loss", loss, prog_bar=True)

        if self.classifier_model is not None:
            baseline_probs = torch.nn.functional.softmax(
                                    classification_output(self.classifier_model,
                                                          start_embedding),
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
                             ) / start_embedding.shape[0]
            dCorrect = ((perturbed_class == labels).sum() - (base_class == labels).sum()
                        ) / start_embedding.shape[0]

            self.log("eval/improved_probability", frac_improved)
            self.log("eval/improved_classification", dCorrect)

    def configure_optimizers(self):
        """Hugging Face-style AdamW with parameter groups excluding bias and
        LayerNorm from weight decay"""
        learning_rate = self.lr
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


class EmbeddingToGradient(BaseEmbeddingRegressor):
    """EmbeddingToGradient: Transformer model that directly maps
    image embeddings to loss function gradients
    """
    def __init__(self, classifier_model=None, step_size=100,
                 num_transformer_layers=6, num_fc_layers=2, **kwargs
                 ):
        super().__init__(classifier_model=classifier_model, step_size=step_size,
                         num_transformer_layers=num_transformer_layers,
                         num_fc_layers=num_fc_layers
                         )

    def next_embedding(self, batch):
        """ Uses model predicted gradient steps to predict the embedding """

        start_embedding, gradient_gt, labels = map(lambda x: x.squeeze(), batch)
        outputs = self.forward(start_embedding, targets=gradient_gt)

        gradient_pred = outputs.predictions
        loss = outputs.loss

        perturbed_embedding = start_embedding - self.step_size * gradient_pred

        return RegressionModelOutput(
                           predictions=perturbed_embedding,
                           starting_point=start_embedding,
                           loss=loss,
                           labels=labels
                          )

    def training_step(self, batch, batchid=None):
        embedding, gradient = map(lambda x: x.squeeze(), batch)
        target = self.step_size * gradient

        loss = self.forward(embedding, target).loss
        self.log("train/loss", loss.item(), on_epoch=False, prog_bar=True)
        return loss


class EmbeddingPropagation(BaseEmbeddingRegressor):
    """EmbeddingPropagation: evolves the Embedding to simulate a gradient step
    """
    def __init__(self, classifier_model=None, step_size=100,
                 num_transformer_layers=6, num_fc_layers=2,
                 lr=5e-5, **kwargs
                 ):
        super().__init__(classifier_model=classifier_model, step_size=step_size,
                         num_transformer_layers=num_transformer_layers,
                         num_fc_layers=num_fc_layers
                         )

    def next_embedding(self, batch):
        """ predicts the embedding values after a gradient step directly"""
        start_embedding, gradient_gt, labels = map(lambda x: x.squeeze(), batch)
        outputs = self.forward(start_embedding, targets=gradient_gt)

        loss = outputs.loss
        perturbed_embedding = outputs.predictions

        return RegressionModelOutput(
                           predictions=perturbed_embedding,
                           starting_point=start_embedding,
                           loss=loss,
                           labels=labels
                          )

    def training_step(self, batch, batchid=None):
        embedding, gradient = map(lambda x: x.squeeze(), batch)

        # Now the target is the embedding after evolution along the gradient
        target = embedding - self.step_size * gradient

        loss = self.forward(embedding, targets=target).loss
        self.log("train/loss", loss.item(), on_epoch=False, prog_bar=True)
        return loss


class ClsTokenPropagation(BaseEmbeddingRegressor):
    """ClsTokenPropagation: evolves the Embedding to simulate a
    gradient step on the class token only
    """
    def __init__(self, classifier_model=None, step_size=100,
                 num_transformer_layers=6, num_fc_layers=2,
                 lr=5e-5, **kwargs
                 ):
        super().__init__(classifier_model=classifier_model, step_size=step_size,
                         num_transformer_layers=num_transformer_layers,
                         num_fc_layers=num_fc_layers
                         )

    def forward(self, embeddings, targets=None):
        """ Applies the models in order, compute loss
        focuses only on the first element of the embedding
        """
        embeddings_copy = embeddings.clone().detach()
        predictions = self.fc_layers(self.transformer_layers(embeddings).last_hidden_state)

        predictions[:, 1:, :] = embeddings_copy[:, 1:, :]

        loss = None
        if targets is not None:
            # mimic the effects of step size in the loss function
            loss = self.loss(predictions, targets)

        return RegressionModelOutput(
                           predictions=predictions,
                           loss=loss
                          )

    def training_step(self, batch, batchid=None):
        embedding, gradient = map(lambda x: x.squeeze(), batch)

        # Now the target is the embedding after evolution along the gradient
        target = embedding - self.step_size * gradient

        loss = self.forward(embedding, targets=target).loss
        self.log("train/loss", loss.item(), on_epoch=False, prog_bar=True)
        return loss

    def next_embedding(self, batch):
        """ predicts the embedding values after a gradient step """

        start_embedding, gradient_gt, labels = map(lambda x: x.squeeze(), batch)
        outputs = self.forward(start_embedding, targets=gradient_gt)

        loss = outputs.loss
        perturbed_embedding = outputs.predictions

        return RegressionModelOutput(
                           predictions=perturbed_embedding,
                           starting_point=start_embedding,
                           loss=loss,
                           labels=labels
                          )


def classification_output(classifier_model: torch.nn.Module, embeddings: torch.Tensor
                          ) -> torch.Tensor: 
    x = classifier_model.classifier.vit.encoder(embeddings).last_hidden_state
    x = classifier_model.classifier.vit.layernorm(x)[:, 0, :]
    logits = classifier_model.classifier.classifier(x)

    return logits
