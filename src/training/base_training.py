# Training scripts for TTA models
import torch
import evaluate
import inspect
import numpy as np

from transformers import Trainer, TrainingArguments


def raw_batch_collator(data):
    return {'pixel_values': torch.stack([ele[0] for ele in data]),
            'labels': torch.stack([ele[1] for ele in data])
            }


def compute_metrics(eval_preds):
    metric = evaluate.load('accuracy')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def probing_trainer(model, train_dataset, val_dataset, freeze_cls_token=True,
                    **kwargs):
    """ Trainer for probing models, that only train the classifier.
        Goal: _good enough_ performance across a range of ViT probing
    """
    model.freeze_embedding(freeze_cls_token)

    kwarg_defaults = {
        'learning_rate': 5E-5,
        'num_train_epochs': 3,

        'weight_decay': 0.01,

        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.05,

        'logging_steps': 50,
        'logging_strategy': "steps",

        'eval_strategy': "epoch",
        'metric_for_best_model': 'eval_accuracy',
        'greater_is_better': True,
        'save_strategy': "best",
        'save_total_limit': 1,
        'load_best_model_at_end': True,

        'output_dir': 'initial_train',
        'dataloader_num_workers': 8,
        'report_to': "wandb",
        'dataloader_persistent_workers': True,
    }

    all_args = {**kwarg_defaults, **kwargs}

    training_args = TrainingArguments(**args_filter(all_args, TrainingArguments))

    trainer = Trainer(model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      data_collator=raw_batch_collator,
                      compute_metrics=compute_metrics,
                      )
    return trainer


def full_trainer_classification(model, train_dataset, val_dataset, **kwargs):
    """ Trainer for full models to perform classification 
    """
    model.unfreeze_all()

    kwarg_defaults = {
        'learning_rate': 5E-5,
        'num_train_epochs': 3,

        'weight_decay': 0.01,

        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.05,

        'logging_steps': 50,
        'logging_strategy': "steps",

        'eval_strategy': "epoch",
        'metric_for_best_model': 'eval_accuracy',
        'greater_is_better': True,
        'save_strategy': "best",
        'save_total_limit': 1,
        'load_best_model_at_end': True,

        'output_dir': 'initial_train',
        'dataloader_num_workers': 8,
        'report_to': "wandb",
        'dataloader_persistent_workers': True,
    }

    all_args = {**kwarg_defaults, **kwargs}

    training_args = TrainingArguments(**args_filter(all_args, TrainingArguments))

    trainer = Trainer(model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      data_collator=raw_batch_collator,
                      compute_metrics=compute_metrics,
                      )
    return trainer


def decoder_synchronization_training(model, train_dataset, val_dataset, **kwargs):
    """ Trainer to synchronize the decoder representation with learned encoder
        represetation to prevent old decoder representations from influencing TTA
        learning
    """

    def sync_collator(data):
        return {'pixel_values': model.preprocess(torch.stack([ele[0] for ele in data]))
                }

    embedding_model = model.embedding

    # freeze / unfreeze parameters
    for parameter in embedding_model.vit.parameters():
        parameter.requires_grad = False
    for parameter in embedding_model.decoder.parameters():
        parameter.requires_grad = True

    kwarg_defaults = {
        'num_train_epochs': 3,

        'logging_steps': 50,
        'logging_strategy': "steps",

        'eval_strategy': "epoch",
        'metric_for_best_model': 'eval_loss',
        'greater_is_better': False,
        'save_strategy': "best",
        'save_total_limit': 1,
        'load_best_model_at_end': True,

        'eval_accumulation_steps': 16,

        'output_dir': 'initial_train',
        'report_to': "wandb",
    }

    all_args = {**kwarg_defaults, **kwargs}

    training_args = TrainingArguments(**args_filter(all_args, TrainingArguments))

    trainer = Trainer(embedding_model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      data_collator=sync_collator,
                      )

    trainer.can_return_loss = True

    return trainer


def args_filter(args, func):
    possible_args = inspect.signature(func).parameters
    return {k: v for k, v in args.items() if k in possible_args}
