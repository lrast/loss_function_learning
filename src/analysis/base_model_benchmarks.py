# Bencahmarks and tests for base models

import re
import evaluate
import gc
import torch

import numpy as np

from tqdm import tqdm
from pathlib import Path
from huggingface_hub.errors import HFValidationError

from src.models.TTA_model import ClassifierWithTTA
from src.data.image_data import balanced_train_subsets, dataset_from_file


def test_accuracy(checkpoint_dir, **kwargs) -> list[dict]:
    """ Step zero analysis of the models that we trained. """
    try:
        with LoadedNetwork(checkpoint_dir) as model:
            # should warn about train error
            test_set, _ = balanced_train_subsets(total_size=10000, split='valid',
                                                 train_fraction=1)
            accuracy = evaluate_accuracy(model, test_set)
            return [accuracy]

    except HFValidationError:
        print(f'no checkpoint available in {checkpoint_dir}')
        return [{}]


def corrupted_accuracy(checkpoint_dir, data_dir='dataset_files', **kwargs
                       ) -> list[dict]:
    """ Accuracy on the corrupted generalization sets """
    try:
        with LoadedNetwork(checkpoint_dir) as model:
            # should warn about train error
            output_rows = []
            for datafile in Path(data_dir).glob('corrupted_*_severity*.npz'):
                corruption_name = re.search(r'corrupted_(.*)_severity', str(datafile)
                                            ).group(1)
                print(f'\n {corruption_name}:')
                test_set = dataset_from_file(filename=datafile, device='mps')
                accuracy = evaluate_accuracy(model, test_set, num_workers=0)
                accuracy.update({'corruption': corruption_name})

                output_rows.append(accuracy)

                del test_set

            return output_rows

    except HFValidationError:
        print(f'no checkpoint available in {checkpoint_dir}')
        return [{}]


def corrupted_accuracy_sample_cls_tokens(checkpoint_dir,
                                         cls_token_dataset,
                                         data_dir='dataset_files',
                                         **kwargs
                                         ) -> list[dict]:
    """ Accuracy on the corrupted generalization sets with cls tokens sampled
    from the previous uncorrupted run data.

    Current written very specifically for the imediate experiment
    """
    model = ClassifierWithTTA.from_pretrained(checkpoint_dir)
    model = model.to('mps')
    model = model.eval()
    module = model.embedding.vit.layernorm

    activity_samples = torch.load(cls_token_dataset)['activity'].view(-1, 768)
    N = activity_samples.shape[0]

    def randomization_hook(module, input, output):
        copied_output = output.clone()
        batch, tokens, dims = copied_output.shape

        cls_tokens = activity_samples[torch.randperm(N)[0:batch]]
        copied_output[:, 0, :] = cls_tokens.to(copied_output.device)

        return copied_output

    # iterate through different corruptions
    output_rows = []
    for datafile in Path(data_dir).glob('*.npz'):
        if datafile.stem == 'uncorrupted_valid':
            corruption_name = 'uncorrupted'
        else:
            corruption_name = re.search(r'corrupted_(.*)_severity', str(datafile)
                                        ).group(1)
        print(f'\n {corruption_name}:')
        test_set = dataset_from_file(filename=datafile, device='mps')

        # raw corrupted
        accuracy_baseline = evaluate_accuracy(model, test_set, num_workers=0)
        accuracy_baseline.update({'corruption': corruption_name, 'randomize': False})

        output_rows.append(accuracy_baseline)

        handle = module.register_forward_hook(randomization_hook)
        accuracy_randomized = evaluate_accuracy(model, test_set, num_workers=0)
        accuracy_randomized.update({'corruption': corruption_name, 'randomize': True})
        output_rows.append(accuracy_randomized)
        handle.remove()

        print("Forward Hooks:", module._forward_hooks)
        print("Forward Pre-Hooks:", module._forward_pre_hooks)

        del test_set

    return output_rows


def TTA_accuracy(checkpoint_dir):
    """ Accuracy on the corrupted generalization sets, with TTA """
    pass


def evaluate_accuracy(model, dataset, device='mps', num_workers=4):
    """Raw accuracy evaluation"""
    model = model.to(device)
    model.eval()

    dl = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=num_workers)

    accuracy_metric = evaluate.load('accuracy')
    for batch in tqdm(iter(dl), desc="Evaluating accuracy"):
        images, labels = batch
        predictions = np.argmax(model(images.to(device)).logits.detach().cpu(), axis=-1)
        accuracy_metric.add_batch(predictions=predictions, references=labels)

    return accuracy_metric.compute()


class LoadedNetwork(object):
    def __init__(self, checkpoint_dir):
        self.model = ClassifierWithTTA.from_pretrained(checkpoint_dir)

    def __enter__(self):
        return self.model

    def __exit__(self, type, value, traceback):
        self.model.to('cpu')
        del self.model
        gc.collect()
        torch.mps.empty_cache()
