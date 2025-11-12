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
