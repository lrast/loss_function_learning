# Custom data management to get around huggingface's poor handling of image data
import datasets
import torch
import multiprocessing
import warnings

import numpy as np

from tqdm import tqdm
from transformers import ViTImageProcessor

from torch.utils.data import TensorDataset, random_split


def balanced_train_subsets(dataset_name='zh-plus/tiny-imagenet',
                           processor_name='facebook/vit-mae-base',
                           total_size=100000,
                           seed=42,
                           train_fraction=0.9,
                           split='train'
                           ):
    """Loads balanced subsets of training data, and returns them as datasets
    in memory.

    Despite the name, I have extended this to also fetch test set data.
    """
    if split != 'train':
        warnings.warn("You are not fetching train data")

    dataset = datasets.load_dataset(dataset_name, split=split)

    processor = ViTImageProcessor.from_pretrained(processor_name,
                                                  do_convert_rgb=True,
                                                  do_normalize=False,
                                                  do_rescale=False,
                                                  do_resize=True
                                                  )

    def process_images_in_mem(dataset, inds):
        if len(inds) == 0:
            return None

        img_np = [processor(dataset[i]['image'])['pixel_values'][0]
                  for i in inds]
        images = torch.from_numpy(np.stack(img_np))
        labels = torch.tensor(dataset.select(inds)['label'])

        return torch.utils.data.TensorDataset(images, labels)

    gen = torch.Generator()
    gen.manual_seed(seed)

    labels = torch.tensor(dataset['label'][:])

    samples_per = total_size // len(labels.unique())
    num_train = int(train_fraction*samples_per)

    train_inds = []
    val_inds = []
    for label in labels.unique():
        locations = torch.where(labels == label)[0]
        loc_i = torch.randperm(len(locations), generator=gen)

        train_inds.extend(locations[loc_i[0:num_train]].tolist())
        val_inds.extend(locations[loc_i[num_train:samples_per]].tolist())

    return process_images_in_mem(dataset, train_inds), process_images_in_mem(dataset, val_inds)


def fetch_dataset_from_hf(dataset_name='zh-plus/tiny-imagenet',
                          processor_name='facebook/vit-mae-base',
                          split=None):
    """ Load huggingface dataset into memory and return corresponding
        pytorch DataSet
    """
    dataset = datasets.load_dataset(dataset_name, split=split)
    processor_initial = ViTImageProcessor.from_pretrained(processor_name,
                                                          do_convert_rgb=True,
                                                          do_normalize=False,
                                                          do_rescale=False,
                                                          do_resize=True
                                                          )

    # avoid using dataset.map for preprocessing: it slows later retrieval.
    def process_images_in_mem(dataset):
        img_np = [processor_initial(dataset[i]['image'])['pixel_values'][0]
                  for i in range(len(dataset))]
        images = torch.from_numpy(np.stack(img_np))
        labels = torch.tensor(dataset['label'])

        return torch.utils.data.TensorDataset(images, labels)

    if split is None:
        return process_images_in_mem(dataset['train']), process_images_in_mem(dataset['valid'])
    else:
        return process_images_in_mem(dataset)


def make_imageset(dataset_name, split):
    """ Load huggingface dataset and save images as numpy files.
    """
    datasets.disable_caching()
    raw_data = datasets.load_dataset(dataset_name, split=split)

    def process_image(image):
        return np.asarray(image.convert('RGB').resize((224, 224)), dtype=np.uint8)

    pool = multiprocessing.Pool(8)

    images = list(tqdm(
                    pool.imap(process_image, raw_data['image']),
                    total=len(raw_data),
                    desc="Processing Data"
                    ))
    pool.close()

    labels = np.array(list(raw_data['label']))
    images = np.stack(images)

    np.savez(f'pixel_processed_{split}',
             image=images, label=labels)


def dataset_from_file(filename='', split=None, device='cpu', pinned=False):
    """ Returns a Tensor dataset on the device requested. """
    data = np.load(filename)
    images = torch.as_tensor(data['image']).permute([0, 3, 1, 2]).to(device)
    labels = torch.as_tensor(data['label']).to(device)

    if pinned:
        images = images.pin_memory()
        labels = labels.pin_memory()

    dataset = TensorDataset(images, labels)

    if split is None:
        return dataset

    return random_split(dataset, split)


def consistent_subset(dataset, count=1000, seed=42):
    """ Sample a consistent subset of images, balanced between indices. """
    generator = torch.Generator()
    generator.manual_seed(42)

    labels = dataset[:][1]
    samples_per = count // len(labels.unique())

    inds = []
    for label in labels.unique():
        locations = torch.where(labels == label)[0]
        loc_i = torch.randperm(len(locations), generator=generator)[0:samples_per]
        inds.extend(locations[loc_i].tolist())

    return torch.utils.data.TensorDataset(dataset[inds][0].clone(), dataset[inds][0].clone())
