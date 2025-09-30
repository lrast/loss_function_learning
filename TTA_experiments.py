# generates TTA curves
import numpy as np

from training import test_time_adaptation
from model import ClassifierWithTTA

from safetensors import safe_open


def TTA_curve(path, dataset, classifier_hidden_layers=2,
              steps=30, evaluate_freq=5,
              device='cuda:0',
              ):
    """ Full TTA curve for a random subset of the dataset
    """
    state_dict = {}
    with safe_open(f"{path}/model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    model = ClassifierWithTTA(classifier_hidden_layers=classifier_hidden_layers)

    results = np.zeros(steps // evaluate_freq + 1)

    for i in range(len(dataset)):
        model.load_state_dict(state_dict)

        inputs = dataset[i][0][None, :, :, :]
        labels = dataset[i][1]

        sample = test_time_adaptation(model, inputs, labels=labels,
                                      steps=steps, evaluate_freq=evaluate_freq,
                                      device=device)
        results += sample

    return results / len(dataset)
