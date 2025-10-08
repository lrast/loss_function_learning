import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from ..data.activity_data import ActivityGradientDataDict


def make_gradient_trainer(classifier_model, gradient_model,
                          images_train, images_val, 
                          project_name='TTA_loss', run_name='debug',
                          checkpoint_directory='debug',
                          epochs=10, step_size=1E-1, **kwargs):
    """Make a pytorch lightgning trainer for the gradient model"""

    default_ds_args = {
                        'batch_size': 8,
                        'shuffle': True,
                        'rand_seed': 42,
                        'device': 'mps'
                        }
    ds_args = {k: kwargs.pop(k, v) for k, v in default_ds_args.items()}
    activity_ds = ActivityGradientDataDict(
        classifier_model, classifier_model.embedding.vit,
        images_train,  images_val,
        **ds_args
    )

    # DataLoaders
    train_loader = DataLoader(activity_ds['train'], batch_size=1)
    val_loader = DataLoader(activity_ds['val'], batch_size=1)

    # Set up PyTorch Lightning Trainer
    default_trainer_args = {
        'max_epochs': epochs,
        'accelerator': 'auto',
        'log_every_n_steps': 20,
        'enable_progress_bar': True,
    }
    trainer_args = {**default_trainer_args, **kwargs}
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_directory,
        save_top_k=1,
        monitor='eval/loss',
        mode='min',
        save_last=True,
        every_n_epochs=1
    )

    wandb_logger = WandbLogger(name=run_name, project=project_name)

    pl_trainer = pl.Trainer(
        **trainer_args,
        logger=[wandb_logger],
        callbacks=[checkpoint]
    )

    return pl_trainer, gradient_model, train_loader, val_loader


def make_propagation_trainer(classifier_model, gradient_model,
                             images_train, images_val, 
                             project_name='TTA_loss', run_name='debug',
                             checkpoint_directory='debug',
                             epochs=10, **kwargs):
    """Make a pytorch lightgning trainer for the propagation model"""

    default_ds_args = {
                        'batch_size': 8,
                        'shuffle': True,
                        'rand_seed': 42,
                        'device': 'mps'
                        }
    ds_args = {k: kwargs.pop(k, v) for k, v in default_ds_args.items()}
    activity_ds = ActivityGradientDataDict(
        classifier_model, classifier_model.embedding.vit,
        images_train,  images_val,
        **ds_args
    )

    # DataLoaders
    train_loader = DataLoader(activity_ds['train'], batch_size=1)
    val_loader = DataLoader(activity_ds['val'], batch_size=1)

    # Set up PyTorch Lightning Trainer
    default_trainer_args = {
        'max_epochs': epochs,
        'accelerator': 'auto',
        'log_every_n_steps': 20,
        'enable_progress_bar': True,
    }
    trainer_args = {**default_trainer_args, **kwargs}
    checkpoint = ModelCheckpoint(
                                dirpath=checkpoint_directory,
                                save_top_k=1,
                                monitor='eval/loss',
                                mode='min',
                                save_last=True,
                                every_n_epochs=1
                            )

    wandb_logger = WandbLogger(name=run_name, project=project_name)

    pl_trainer = pl.Trainer(
        **trainer_args,
        logger=[wandb_logger],
        callbacks=[checkpoint]
    )

    return pl_trainer, gradient_model, train_loader, val_loader
