# Train script for baseline models that perform TTA
import hydra
import gc
import torch
import os
import wandb

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.models.TTA_model import ClassifierWithTTA
from src.data.image_data import balanced_train_subsets
from src.training.TTA_training import probing_trainer, full_trainer_classification, \
                                       decoder_synchronization_training
from pathlib import Path


@hydra.main(config_path="../configs/TTA_train", config_name="base", version_base="1.2")
def main(cfg: DictConfig):
    # Initialize underlying model and data
    images_train, images_val = balanced_train_subsets(**cfg.data)

    if cfg.debug.train_as_validation:
        images_val = images_train

    def train_and_cleanup(get_trainer, input_dir, output_dir,
                          train_cfg
                          ):
        """ Initialize the model, train it, and clean-up """

        if input_dir is None:
            model = ClassifierWithTTA(**cfg.model)
        else:
            model = ClassifierWithTTA.from_pretrained(input_dir / 'best', **cfg.model)

        trainer = get_trainer(model, images_train, images_val,
                              output_dir=output_dir,
                              **train_cfg)
        trainer.train()

        best_model_dir = os.path.join(output_dir, "best")
        trainer.save_model(best_model_dir)
        wandb.finish()

        # Cleanup
        model.to('cpu')
        del model
        gc.collect()
        try:
            torch.mps.empty_cache()
        except:
            pass
        try:
            torch.cuda.empty_cache()
        except:
            pass

    base_directory = Path(HydraConfig.get().runtime.output_dir) / 'checkpoints/'
    working_directory = None

    # Run the training steps
    if not cfg.train.probe.disable:
        # Linear probing
        probe_directory = base_directory / 'probe'
        train_and_cleanup(probing_trainer, working_directory, probe_directory,
                          cfg.train.probe)
        working_directory = probe_directory

    if not cfg.train.full.disable:
        # Fine tune base
        full_directory = base_directory / 'full'
        train_and_cleanup(full_trainer_classification, working_directory, full_directory,
                          cfg.train.full)
        working_directory = full_directory

    if not cfg.train.decoder_sync.disable:
        # Sync decoder
        sync_directory = base_directory / 'sync'
        train_and_cleanup(decoder_synchronization_training, working_directory, sync_directory,
                          cfg.train.decoder_sync)


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "TTA_loss"
    main()
