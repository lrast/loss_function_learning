# script for internal activity analyzer training and data creation

import torch

from tqdm import tqdm
from pathlib import Path

from src.data.activity_data import CLSEmbeddingDataset
from src.data.image_data import balanced_train_subsets

from src.models.TTA_model import ClassifierWithTTA
from src.models.activity_analysis import DeepSets, DeepSetsFoundation, make_trainer


from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import hydra


# Starting with configurations stored locally
default_config = {
    "mode": 'data_gen',
    "data": {
        "split": "val",
        "train_fraction": 0.9,
        "file_name": 'activity_data.pt',
        "shuffle": True
    },
    "model": {
        "type": 'deepsets',
        "ckpt_dir": '',
        "params": {},
    },
    "train": {
        "batch_size": 100,
        "preprocess": None
    },
    "hydra": {"run": {"dir": 'activity_analysis/${now:%Y-%m-%d_%H_%M_%S}_${mode}'}}
}

cs = ConfigStore.instance()
cs.store(name="base_config", node=default_config)


@hydra.main(version_base=None, config_name="base_config")
def main(cfg: DictConfig) -> None:
    if cfg.mode == 'data_gen':
        # activity samples generation mode
        print('loading images...')

        if cfg.data.split in ['train', 'val', 'valid', 'eval']:
            datasets = balanced_train_subsets(train_fraction=cfg.data.train_fraction, split='train')
            image_data = (datasets[0] if cfg.data.split == 'train' else datasets[1])

        elif cfg.data.split == 'test':
            image_data, _ = balanced_train_subsets(train_fraction=cfg.data.train_fraction, split='valid')

        else:
            raise ValueError(f'Unknown split {cfg.data.split}')

        full_model = ClassifierWithTTA.from_pretrained(cfg.model.ckpt_dir)

        activity_dataset = CLSEmbeddingDataset(full_model, image_data, return_labels=True,
                                               device='mps', shuffle=cfg.data.shuffle)

        activity = []
        predictions = []
        gts = []

        data_file = Path(HydraConfig.get().runtime.output_dir) / cfg.data.file_name

        def save_current(data_file, **kwargs):
            torch.save({k: torch.concat(v) for k, v in kwargs.items()}, data_file)

        for i, batch in tqdm(enumerate(iter(activity_dataset)),
                             total=len(image_data)//8, desc='generating data'):
            activity.append(batch[0])
            gts.append(batch[1])
            predictions.append(batch[2])

            if i % 100 == 0 and i > 0:
                save_current(data_file, activity=activity, predictions=predictions, gts=gts)

        save_current(data_file, activity=activity, predictions=predictions, gts=gts)

    # single training run model
    elif cfg.mode == 'train':
        model_registry = {'deepsets': DeepSets,
                          'deepsetsfoundation': DeepSetsFoundation}

        def filter_activity(data_dict, mode='correct_single'):
            if mode != 'correct_single':
                raise NotImplementedError('Other filters not implemented')

            correct_preds = data_dict['predictions'].mode(1).values == data_dict['gts']
            single_counts = torch.tensor([len(row.unique()) == 1
                                         for row in data_dict['predictions']])

            activity = data_dict['activity'][correct_preds & single_counts]
            correct_outs = correct_preds[correct_preds & single_counts].to(torch.float)
            return activity, correct_outs

        data_dict = torch.load(cfg.data.file_name)

        if cfg.train.preprocess is None:
            activity = data_dict['activity']
            correct_outs = (data_dict['predictions'].mode(1).values == data_dict['gts']).to(torch.float)
        else:
            activity, correct_outs = filter_activity(data_dict, cfg.train.preprocess)

        full_ds = torch.utils.data.TensorDataset(activity, correct_outs)

        activityAnalyzer = model_registry[cfg.model.type.lower()](**cfg.model.params)

        ckpt_file = Path(HydraConfig.get().runtime.output_dir) / cfg.model.ckpt_dir
        trainer = make_trainer(ckpt_file)

        train_ds, val_ds = torch.utils.data.random_split(full_ds, (0.95, 0.05))
        train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True,
                                               batch_size=cfg.train.batch_size)
        val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False,
                                             batch_size=cfg.train.batch_size)

        trainer.fit(activityAnalyzer, train_dl, val_dl)

    elif cfg.mode == 'train_sweep':
        pass


if __name__ == "__main__":
    main()
