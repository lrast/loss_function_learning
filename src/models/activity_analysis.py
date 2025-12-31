# Internal activity analysis
import torch
import wandb

import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.metrics import f1_score


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
                 weight_decay=0.05,
                 loss_weights=[1., 1.],
                 pooling='mean',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.pooling = pooling

        # φ : per-element encoder
        self.phi = MLP([input_dim] + hidden_phi, dropout_p=phi_dropout)

        # ρ : classifier on pooled representation
        self.rho = MLP([hidden_phi[-1]] + hidden_rho + [2], dropout_p=rho_dropout)

        self.loss = nn.CrossEntropyLoss(weight=torch.tensor(loss_weights))

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
        """ validation metrics should reflect:
            - overall accuracy
            - accuracy on each 
            - detection success on negatives
        """
        x, correct_classification = batch

        predictions = self.forward(x)
        correctness_prediction = predictions.argmax(1)

        accuracy, accuracy_correct, accuracy_incorrect, \
            incorrect_f1 = metrics(correctness_prediction.detach().cpu(),
                                   correct_classification.detach().cpu())

        self.log('eval/accuracy', accuracy)
        self.log('eval/accuracy_correct', accuracy_correct)
        self.log('eval/accuracy_incorrect', accuracy_incorrect)
        self.log('eval/F1_incorrect', incorrect_f1)

    def test_step(self, batch, batch_idx):
        x, correct_classification = batch

        predictions = self.forward(x)
        correctness_prediction = predictions.argmax(1)

        accuracy, accuracy_correct, accuracy_incorrect, \
            incorrect_f1 = metrics(correctness_prediction, correct_classification)

        self.log('test/accuracy', accuracy)
        self.log('test/accuracy_correct', accuracy_correct)
        self.log('test/accuracy_incorrect', accuracy_incorrect)
        self.log('test/F1_incorrect', incorrect_f1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3,
                                weight_decay=self.hparams.weight_decay)


class DeepSetsFoundation(pl.LightningModule):
    """ DeepSets-like model (in that it learns sufficient statistics)
    Trained by contrastive training on the sufficient statistics

    Note that this likely requires larger batch sizes
    """
    def __init__(self, 
                 input_dim=768, 
                 hidden_phi=[256, 128], 
                 phi_dropout=0.1,
                 train_key_ratio=0.5,
                 temperature=0.07,

                 weight_decay=1E-2,
                 lr=1E-4,
                 **kwargs
                 ):
        super().__init__()
        # φ : per-element encoder
        self.save_hyperparameters()

        self.phi = MLP([input_dim] + hidden_phi, dropout_p=phi_dropout)

        # How to divide the samples into query and key sets
        self.train_key_ratio = train_key_ratio
        self.temperature = temperature

    def forward(self, x):
        """
        apply φ-network to each element independently
        """

        phi_vals = self.phi(x)  # (batch, samples, dims)
        return phi_vals.mean(1)

    def _shared_eval_step(self, batch, batch_idx):
        """ Contrastive training """
        samples, _ = batch

        batch_size, num_samples, dim = samples.shape

        # randomly partition the tensor along the sample dimension
        perms = torch.stack([torch.randperm(num_samples, device=self.device)
                             for _ in range(batch_size)])

        keys_count = int(num_samples * self.train_key_ratio)
        idx_a = perms[:, :keys_count]
        idx_b = perms[:, keys_count:]

        samples_a = torch.gather(samples, 1, idx_a.unsqueeze(-1).expand(-1, -1, dim))
        samples_b = torch.gather(samples, 1, idx_b.unsqueeze(-1).expand(-1, -1, dim))

        # evaluate on both
        z_a = self.forward(samples_a)
        z_b = self.forward(samples_b)

        # normalize features
        z_a = torch.nn.functional.normalize(z_a, p=2, dim=1)
        z_b = torch.nn.functional.normalize(z_b, p=2, dim=1)

        # classify which batch I came from
        logits = z_a @ z_b.T
        targets = torch.arange(batch_size, device=self.device)

        loss = torch.nn.functional.cross_entropy(logits / self.temperature,
                                                 targets)

        return loss, logits, targets

    def training_step(self, batch, batch_idx=None):
        loss, _, _ = self._shared_eval_step(batch, batch_idx)

        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx=None):
        """ validation metrics should reflect:
            - overall accuracy
            - accuracy on each 
            - detection success on negatives
        """
        batch_size = batch[0].shape[0]
        loss, logits, targets = self._shared_eval_step(batch, batch_idx)

        with torch.no_grad():
            # Calculate Top-1 Accuracy
            preds = torch.argmax(logits, dim=1)
            acc = (preds == targets).float().mean()
            
            # Calculate 'Collapse' Monitor
            # (High similarity between DIFFERENT items in the batch is bad)
            mask = ~torch.eye(batch_size, device=self.device).bool()
            avg_neg_sim = logits[mask].mean()

        self.log("eval/accuracy", acc)
        self.log("eval/negative_similarity", avg_neg_sim)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        
        # inear warmup for the first 500 steps
        def lr_lambda(current_step):
            if current_step < 500:
                return float(current_step) / float(500)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # important to update every step, not every epoch
            },
        }


def metrics(prediction, ground_truth):
    accuracy = (prediction == ground_truth).to(int).sum() / len(ground_truth)

    accuracy_correct = prediction[ground_truth == 1].float().mean()
    accuracy_incorrect = (1. - prediction[ground_truth == 1]).mean()

    incorrect_f1 = f1_score(1 - ground_truth, 1-prediction)

    return accuracy, accuracy_correct, accuracy_incorrect, incorrect_f1 


def make_trainer(dir_path):
    # weighted sampling of train data
    #class_sample_counts = np.bincount(targets)
    #weights = 1. / torch.tensor(class_sample_counts / class_sample_counts.sum(), dtype=torch.float)
    #sample_weights = weights[targets.to(int)]

    #sampler = torch.utils.data.WeightedRandomSampler(
    #                                                  sample_weights, 
    #                                                  num_samples=len(sample_weights),
    #                                                  replacement=True
    #                                                )

    wandb.init(project='TTA_loss', group='activity_analysis')
    wandb_logger = WandbLogger(project="TTA_loss")

    checkpoint_callback = ModelCheckpoint(dirpath=dir_path, save_last=True)

    trainer = pl.Trainer(max_epochs=30, logger=wandb_logger,
                         callbacks=[checkpoint_callback])

    return trainer


def make_dataloaders(activity_file):
    """ """
    pass
