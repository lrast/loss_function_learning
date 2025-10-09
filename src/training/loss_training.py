import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def make_trainer(max_epochs=10, run_name='debug', dirpath='debug'):
    """"Set up PyTorch Lightning Trainer for loss function models"""
    trainer_args = {
        'max_epochs': max_epochs,
        'accelerator': 'auto',
        'log_every_n_steps': 20,
        'enable_progress_bar': True,
    }

    checkpoint = ModelCheckpoint(
        dirpath=dirpath,
        save_top_k=1,
        monitor='eval/loss',
        mode='min',
        save_last=True,
        filename='best',
        every_n_epochs=1
    )

    wandb_logger = WandbLogger(project='TTA_loss', name=run_name,)

    pl_trainer = pl.Trainer(
        **trainer_args,
        logger=[wandb_logger],
        callbacks=[checkpoint]
    )

    return pl_trainer
