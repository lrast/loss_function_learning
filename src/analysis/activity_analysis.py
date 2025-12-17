# Internal activity analysis
import sys
import numpy as np
import torch
import torch.nn as nn

import pytorch_lightning as pl
import wandb

from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.activity_data import CLSEmbeddingDataset
from src.data.image_data import balanced_train_subsets

from src.models.TTA_model import ClassifierWithTTA


class MLP(nn.Module):
    def __init__(self, dims, activation=nn.ReLU, dropout_p=0.5):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation())
                layers.append(nn.Dropout(p=dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepSets(pl.LightningModule):
    def __init__(self, 
                 input_dim=768, 
                 hidden_phi=[256, 128], 
                 phi_dropout=0.1,
                 hidden_rho=[128, 64],
                 rho_dropout=0.7,
                 pooling='mean'):
        super().__init__()
        self.save_hyperparameters()
        self.pooling = pooling

        # φ : per-element encoder
        self.phi = MLP([input_dim] + hidden_phi, dropout_p=phi_dropout)

        # ρ : classifier on pooled representation
        self.rho = MLP([hidden_phi[-1]] + hidden_rho + [1], dropout_p=rho_dropout)

        self.loss = nn.BCELoss()

    def forward(self, x):
        """
        x: (batch, num_elems=50, input_dim=700)
        """
        # φ-network: apply to each element independently
        phi_x = self.phi(x)   # (B, 50, H)

        # Permutation-invariant pooling
        if self.pooling == 'mean':
            pooled = phi_x.mean(dim=1)
        elif self.pooling == 'sum':
            pooled = phi_x.sum(dim=1)
        elif self.pooling == 'max':
            pooled, _ = phi_x.max(dim=1)
        else:
            raise ValueError("Unknown pooling")

        # ρ-network
        out = self.rho(pooled).squeeze(-1)
        return torch.sigmoid(out)

    def training_step(self, batch, batch_idx=None):
        x, correct_classification = batch
        predictions = self.forward(x)

        loss = self.loss(predictions, correct_classification)
        self.log('train/loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx=None):
        x, correct_classification = batch

        predictions = self.forward(x)
        correctness_prediction = (predictions > 0.5).to(int)

        accuracy = (correctness_prediction == correct_classification
                    ).to(int).sum() / len(correct_classification)

        self.log('eval/accuracy', accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    # train the model 
    mode = 'train'
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == 'data_gen' or mode == 'data_gen_test' or mode == 'data_gen_val':
        # generate activity samples for training 
        ckpt_dir = Path('experiments_base/sweeps/2025-11-13(16:04:10)-base_training/1/checkpoints')
        full_model = ClassifierWithTTA.from_pretrained(ckpt_dir / 'full/best')

        print('loading images...')
        if mode == 'data_gen':
            # full training set
            image_data, _ = balanced_train_subsets(train_fraction=1.0, split='train')
        elif mode == 'data_gen_test':
            # test set for image classifier
            image_data, _ = balanced_train_subsets(train_fraction=1.0, split='valid')
        elif mode == 'data_gen_val':
            # validation set for image classifier
            _, image_data = balanced_train_subsets(train_fraction=0.9)

        print(len(image_data))

        activity_dataset = CLSEmbeddingDataset(full_model, image_data, return_labels=True, device='mps')

        activity = []
        predictions = []
        gts = []

        for i, batch in tqdm(enumerate(iter(activity_dataset)),
                             total=len(image_data)//8, desc='generating data'):
            activity.append(batch[0])
            gts.append(batch[1])
            predictions.append(batch[2])

            if i % 100 == 0 and i > 0:
                torch.save({'activity': torch.concat(activity),
                            'predictions': torch.concat(predictions),
                            'gts': torch.concat(gts)}, 'activity_data.pt')

        torch.save({'activity': torch.concat(activity),
                    'predictions': torch.concat(predictions),
                    'gts': torch.concat(gts)}, 'activity_data.pt')

    else:
        # running the training
        data_dict = torch.load('./activity_data.pt')

        correct_outs = (data_dict['predictions'].mode(1).values == data_dict['gts']).to(torch.float)
        full_ds = torch.utils.data.TensorDataset(data_dict['activity'], correct_outs)

        train_ds, val_ds = torch.utils.data.random_split(full_ds, (0.95, 0.05))

        # weighted sampling of train data
        targets = train_ds[:][1]
        class_sample_counts = np.bincount(targets)
        weights = 1. / torch.tensor(class_sample_counts / class_sample_counts.sum(), dtype=torch.float)
        sample_weights = weights[targets.to(int)]

        sampler = torch.utils.data.WeightedRandomSampler(
                                                          sample_weights, 
                                                          num_samples=len(sample_weights),
                                                          replacement=True
                                                        )

        train_dl = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=32)
        val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False, batch_size=32)
        activityAnalyzer = DeepSets(phi_dropout=0.0, rho_dropout=0.5)
        print(activityAnalyzer.phi.net[2].p)

        wandb.init(project='TTA_loss', name='activity_analysis')
        wandb_logger = WandbLogger(project="TTA_loss")

        checkpoint_callback = ModelCheckpoint(
            dirpath=".",
            monitor="eval/accuracy",
            mode="max",
            save_top_k=1
        )

        trainer = pl.Trainer(max_epochs=30, logger=wandb_logger,
                             callbacks=[checkpoint_callback])
        trainer.fit(activityAnalyzer, train_dl, val_dl)

        torch.save(activityAnalyzer.state_dict(), 'model_weights.pt')
