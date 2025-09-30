# models to capture a surrogate for classification loss.

import torch
import weakref

from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTEncoder

from torch.utils.data import IterableDataset

from dataclasses import dataclass
from typing import Optional
from transformers.utils import ModelOutput


@dataclass
class GradientModelOutput(ModelOutput):
    """
    Base class for outputs of the gradient models
    """
    predictions: torch.FloatTensor = None 
    loss: Optional[torch.FloatTensor] = None


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

        self.local_gradient_approximation = ViTEncoder(config)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels=None):
        raw_outputs = self.local_gradient_approximation(inputs)

        loss = None
        if labels is not None:
            loss = self.loss(raw_outputs.last_hidden_state, labels)

        return GradientModelOutput(
                                   predictions=raw_outputs.last_hidden_state,
                                   loss=loss
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
    def __init__(self, model, module, train_data, val_data=None, device=None):
        recorder = BatchRecorder(module)
        self.datasets = {
            "train": ActivityGradientDataset(model, recorder, train_data,
                                             device=device),
        }
        if val_data is not None:
            self.datasets["val"] = ActivityGradientDataset(model, recorder, val_data,
                                                           device=device)

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

            yield self.recorder.batch
