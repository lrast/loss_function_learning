# models built on huggingface transformers backbones

import torch

from transformers import AutoModelForImageClassification, ViTConfig, \
                         ViTMAEForPreTraining, ViTImageProcessor
from transformers.modeling_outputs import ImageClassifierOutput

from huggingface_hub import PyTorchModelHubMixin

from torchvision.transforms.v2 import Normalize, Compose, ToDtype
from safetensors import safe_open
from typing import Optional


class CustomMAE(ViTMAEForPreTraining):
    """ MAE wrapped that performs
        (TODO) custom loss functions

        Important note: requires preprocessed images.
    """
    def __init__(self, *args, randomized_CLS=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.randomized_CLS = randomized_CLS

    def forward(self, pixel_values):
        """ Full encode-decode cycle on the images """
        outputs = super().forward(pixel_values)
        # add auxilliary loss functions here
        return outputs

    def embedding(self, pixel_values):
        """ Run just the forward embedding step of the MAE """
        self.disable_masking()

        if not self.randomized_CLS:
            return self.vit(pixel_values).last_hidden_state
        else:
            # we need to dig into the guts of the model to randomize cls tokens
            dtype = pixel_values.dtype
            patch_embeddings = self.vit.embeddings.patch_embeddings(pixel_values)
            batch_size, _, hidden_size = patch_embeddings.shape

            if self.randomized_CLS == 'debug':
                default_token = self.vit.embeddings.cls_token.data
                cls_token = default_token.repeat((batch_size, 1, 1))

            else:
                cls_token = torch.randn(batch_size, 1, hidden_size,
                                        device=self.device, dtype=dtype)

            embedding = torch.cat((cls_token, patch_embeddings), 1)
            position_embeddings = self.vit.embeddings.position_embeddings

            embedding = embedding + position_embeddings
            embedding = self.vit.encoder(embedding)
            embedding = self.vit.layernorm(embedding.last_hidden_state)
            return embedding

    def disable_masking(self):
        self.vit.config.mask_ratio = 0.0

    def enable_masking(self, mask_ratio=0.75):
        self.vit.config.mask_ratio = mask_ratio


class ClassifierWithTTA(torch.nn.Module, PyTorchModelHubMixin):
    """ 
        Classifier model with embedding for TTA.
            The Embedding is from an MAE model, while the classifier places 
            classifier_hidden_layers number of ViT layers on top of that 
        
        Used as follows:
             1. This model is trained and used as a classifier.
             2. TTA is performed by training the embedding model directly.
             3. Both models expect torch tensors pixel encoded as uint8
    """
    def __init__(self, classifier_hidden_layers=2, randomized_CLS=False,
                 **kwargs):
        super().__init__()

        # Processor for the vit model
        processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base",
                                                      do_convert_rgb=False,
                                                      do_normalize=True,
                                                      do_rescale=True,
                                                      do_resize=False,
                                                      use_fast=True
                                                      )

        self.preprocess = make_online_transform(processor)

        # Embedding model
        self.embedding = CustomMAE.from_pretrained("facebook/vit-mae-base",
                                                   randomized_CLS=randomized_CLS,
                                                   **kwargs)

        class_config = ViTConfig.from_pretrained("google/vit-base-patch16-224",
                                                 num_hidden_layers=classifier_hidden_layers,
                                                 num_labels=200,
                                                 **kwargs
                                                 )
        self.classifier = AutoModelForImageClassification.from_config(class_config)
        del self.classifier.vit._modules['embeddings']

    def forward(self, pixel_values, labels=None, **kwargs):
        pixel_values = self.preprocess(pixel_values)
        x = self.embedding.embedding(pixel_values)
        x = self.classifier.vit.encoder(x).last_hidden_state
        x = self.classifier.vit.layernorm(x)[:, 0, :]
        logits = self.classifier.classifier(x)
        
        loss = None
        if labels is not None:
            loss = self.classifier.loss_function(labels, logits, self.classifier.config, **kwargs)

        return ImageClassifierOutput(loss=loss, logits=logits)

    def classify(self, pixel_values):
        return self.forward(pixel_values).logits.argmax(1)

    def disable_masking(self):
        self.embedding.disable_masking()

    def enable_masking(self, mask_ratio=0.75):
        self.embedding.enable_masking(mask_ratio)

    def freeze_embedding(self, freeze_cls_token=True):
        for parameter in self.embedding.parameters():
            parameter.requires_grad = False

        if not freeze_cls_token:
            self.embedding.vit.embeddings.cls_token.requires_grad = True

    def unfreeze_all(self):
        for parameter in self.parameters():
            parameter.requires_grad = True

    def load(self, path: str, device: Optional[str] = None) -> None:
        """
        Load model weights from a file
        
        Args:
            path (str): Path to the weights file
            device (str, optional): Device to map the loaded weights to
        """
        try:
            # Load the state dict from the file
            state_dict = {}
            with safe_open(f'{path}/model.safetensors', framework="pt", device=device) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

            # Load the state dict into the model
            self.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {path}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Weights file not found: {path}")
        except Exception as e:
            raise RuntimeError(f"Error loading weights from {path}: {str(e)}")
    
    @classmethod
    def load_from_file(cls, path: str, classifier_hidden_layers: int = 2,
                       device: Optional[str] = None,
                       ):
        """
        Class method to create a model instance and load weights from a file.
        This allows calling NeuralNetworkModel.load_from_file(path) without creating an instance first.
        
        Args:
            path (str): Path to the weights file
            map_location (str, optional): Device to map the loaded weights to

            
        Returns:
            NeuralNetworkModel: Model instance with loaded weights
        """
        model = cls(classifier_hidden_layers)
        model.load(path, device)
        
        return model


def make_online_transform(transform):
    equivalent = Compose([
                         ToDtype(torch.float32, scale=True),
                         Normalize(mean=transform.image_mean, std=transform.image_std)
                         ])
    return equivalent
