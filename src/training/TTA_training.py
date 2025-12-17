# Adaptation be test-time training 
import numpy as np

from transformers import Trainer, TrainingArguments


def test_time_adaptation(model, inputs, labels=None, 
                         repeats=128, steps=20, evaluate_freq=20,
                         device='cuda:0', mask_ratio=0.75,
                         **kwargs):
    """
        Trainer for TTA: low level trainer optimized for speed.

        To do:
            1. make this work with CLS token randomization
            2. add lora version
    """
    # setup
    if f'{model.embedding.device}' != device:
        model = model.to(device)

    embedding_model = model.embedding
    embedding_model.enable_masking(mask_ratio)
    for parameter in embedding_model.parameters():
        parameter.requires_grad = True

    train_data = model.preprocess(inputs)
    inputs = inputs.to(device)
    train_data = train_data.to(device)

    if train_data.shape[0] == 1:
        train_data = train_data.expand([repeats, 3, 224, 224])
    else:
        # multiple input points
        train_data = train_data.expand([repeats, train_data.shape[0], 3, 224, 224])
        train_data = train_data.reshape(-1, 3, 224, 224)

    def run_eval():
        """ Model accuracy evaluation  """
        model.disable_masking()

        preds = model.classify(inputs)
        accuracy = (preds == labels).sum().item() / inputs.shape[0]

        model.enable_masking(mask_ratio)
        return accuracy

    if labels is not None:
        # setup our evaluations 
        labels = labels.to(device)

        num_evals = steps // evaluate_freq + 1
        results = np.zeros(num_evals)
        results[0] = run_eval()

    # train loop
    opt = make_hf_optimizer(embedding_model, **kwargs)
    for i in range(1, steps+1):
        # train
        opt.zero_grad()
        loss = embedding_model(train_data).loss
        loss.backward()
        opt.step()

        # evaluate
        if labels is not None and i % evaluate_freq == 0:
            results[i // evaluate_freq] = run_eval()

    if labels is not None:
        return results


def make_hf_optimizer(model, **kwargs):
    training_args = TrainingArguments(**kwargs)

    optimizer = Trainer(model,
                        args=training_args,
                        ).create_optimizer()
    return optimizer
