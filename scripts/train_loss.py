# Train script for models that learn loss functions of base TTA models
import hydra

from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.models.TTA_model import ClassifierWithTTA
from src.models.loss_models import EmbeddingToGradient, EmbeddingPropagation

from src.data.image_data import balanced_train_subsets
from src.data.activity_data import ActivityGradientDataDict

from src.training.loss_training import make_trainer


MODEL_REGISTRY = {
    "gradient": EmbeddingToGradient,
    "propagation": EmbeddingPropagation,
}


@hydra.main(config_path="../configs/loss_train", config_name="base", version_base="1.2")
def main(cfg: DictConfig):
    # Initialize underlying model and data
    images_train, images_val = balanced_train_subsets(**cfg.image_data)

    if cfg.debug.train_as_validation:
        images_val = images_train

    classifier_model = ClassifierWithTTA.load_from_file(cfg.base_model.ckpt)

    # Initialize loss model and activity dataset
    loss_model = MODEL_REGISTRY[cfg.loss_model._name_](classifier_model=classifier_model,
                                                       **cfg.loss_model.params)
    activity_ds = ActivityGradientDataDict(classifier_model,
                                           classifier_model.embedding.vit,
                                           images_train, images_val,
                                           **cfg.activity_data
                                           )
    train_dl = DataLoader(activity_ds['train'], batch_size=1)
    val_dl = DataLoader(activity_ds['val'], batch_size=1)

    # Initialize trainer
    trainer = make_trainer(dirpath=HydraConfig.get().runtime.output_dir+'/checkpoints/',
                           **cfg.train)

    # Train
    trainer.fit(loss_model, train_dl, val_dl)


if __name__ == "__main__":
    main()
