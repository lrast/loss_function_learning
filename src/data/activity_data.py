# dataset made from network activity, gradient pairs

import weakref
import torch

from torch.utils.data import IterableDataset


class CLSEmbeddingDataset(IterableDataset):
    """Dateset of CLS token embeddings
    """
    def __init__(self, classification_model, raw_inputs,
                 layer_name='embedding.vit.layernorm',
                 repeats=50, return_labels=True,
                 batch_size=8, shuffle=True, seed=42, device=None
                 ):
        super(ActivityGradientDataset).__init__()

        self.classification_model = classification_model.to(device)
        self.raw_inputs = raw_inputs

        self.repeats = repeats
        self.return_labels = return_labels

        # Setup activity hooks
        self.CLS_tokens = None
        module_dict = {k: v for k, v in classification_model.named_modules()}
        self.hook_handle = module_dict[layer_name].register_forward_hook(self.recording_hook)

        # Batching parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

    def recording_hook(self, module, input, output):
        self.CLS_tokens = output[:, 0, :].detach().clone().cpu()

    def __iter__(self):
        if self.shuffle:
            inds = torch.randperm(len(self.raw_inputs), generator=self.gen)
        else:
            inds = torch.arange(len(self.raw_inputs))

        batches = [inds[i: i+self.batch_size].tolist() 
                   for i in range(0, len(self.raw_inputs), self.batch_size)
                   ]

        for batch_inds in batches:
            images, labels = self.raw_inputs[batch_inds]
            if self.repeats == 1:
                preds = self.classification_model.forward(images.to(self.device)
                                                          ).logits.argmax(1)
                activity = self.CLS_tokens

            else:  # loop through the images individually, batching each repeat
                preds = []
                activity = []

                for i in range(images.shape[0]):
                    repeated = images[i: i+1].expand(self.repeats, -1, -1, -1)

                    curr_preds = self.classification_model.forward(repeated.to(self.device)
                                                                   ).logits.argmax(1)

                    activity.append(self.CLS_tokens[None, :])
                    preds.append(curr_preds[None, :])

                activity = torch.cat(activity)
                preds = torch.cat(preds)

            if self.return_labels:
                yield activity, labels.cpu(), preds.detach().cpu()
            else:
                yield activity

    def __del__(self):
        """Cleanup the hooks"""
        self.hook_handle.remove()
        del self.CLS_tokens


class ActivityGradientDataDict:
    """Activity-gradient datasets: training and validation
    """
    def __init__(self, classification_model, module, train_data, val_data=None, **kwargs):
        # Initialize classification_model hooks
        self.recorder = BatchGradientRecorder(module)
        self.datasets = {
            "train": ActivityGradientDataset(classification_model, self.recorder,
                                             train_data, return_labels=False,
                                             **kwargs),
        }
        if val_data is not None:
            self.datasets["val"] = ActivityGradientDataset(classification_model,
                                                           self.recorder,
                                                           val_data,
                                                           return_labels=True,
                                                           **kwargs)

    def __getitem__(self, key):
        return self.datasets[key]


class BatchGradientRecorder:
    """BatchGradientRecorder: records input, gradient pairs for the module specified"""
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


class ActivityGradientDataset(IterableDataset):
    """Dateset of activity and loss function gradients for a given module
    classification_model: parent classification_model containing module of interest
    recorder: an activity / gradient recorder object
    raw_inputs: inputs to parent classification_model to iterate over
    """
    def __init__(self, classification_model, recorder, raw_inputs,
                 batch_size=8, shuffle=True, seed=42,
                 device=None, return_labels=False
                 ):
        super(ActivityGradientDataset).__init__()

        self.classification_model = classification_model.to(device)
        self.raw_inputs = raw_inputs

        self.recorder = recorder

        # Batch parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.return_labels = return_labels

        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            inds = torch.randperm(len(self.raw_inputs), generator=self.gen)
        else:
            inds = torch.arange(len(self.raw_inputs))

        batches = [inds[i: i+self.batch_size].tolist() 
                   for i in range(0, len(self.raw_inputs), self.batch_size)
                   ]

        with torch.enable_grad():
            for batch_inds in batches:
                # Forward and backward passes through the network
                images, labels = self.raw_inputs[batch_inds]
                outs = self.classification_model.forward(images.to(self.device),
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
