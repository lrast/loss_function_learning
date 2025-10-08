# dataset made from network activity, gradient pairs

import weakref
import torch

from torch.utils.data import IterableDataset


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
    def __init__(self, classication_model, module, train_data, val_data=None, **kwargs):
        # Initialize classication_model hooks
        self.recorder = BatchRecorder(module)
        self.datasets = {
            "train": ActivityGradientDataset(classication_model, self.recorder,
                                             train_data, return_labels=False,
                                             **kwargs),
        }
        if val_data is not None:
            self.datasets["val"] = ActivityGradientDataset(classication_model,
                                                           self.recorder,
                                                           val_data,
                                                           return_labels=True,
                                                           **kwargs)

    def __getitem__(self, key):
        return self.datasets[key]


class ActivityGradientDataset(IterableDataset):
    """Dateset of activity and loss function gradients for a given module
    classication_model: parent classication_model containing module of interest
    recorder: an activity / gradient recorder object
    raw_inputs: inputs to parent classication_model to iterate over
    """
    def __init__(self, classication_model, recorder, raw_inputs,
                 batch_size=8, shuffle=True, rand_seed=42,
                 device=None, return_labels=False
                 ):
        super(ActivityGradientDataset).__init__()

        self.classication_model = classication_model.to(device)
        self.raw_inputs = raw_inputs

        self.recorder = recorder

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
                outs = self.classication_model.forward(images.to(self.device),
                                                       labels=labels.to(self.device))
                outs.loss.backward()

                if self.return_labels:
                    yield (*self.recorder.batch, labels)
                else:
                    yield self.recorder.batch


def classification_output(classifier_model: torch.nn.Module, embeddings: torch.Tensor
                          ) -> torch.Tensor: 
    x = classifier_model.classifier.vit.encoder(embeddings).last_hidden_state
    x = classifier_model.classifier.vit.layernorm(x)[:, 0, :]
    logits = classifier_model.classifier.classifier(x)

    return logits
