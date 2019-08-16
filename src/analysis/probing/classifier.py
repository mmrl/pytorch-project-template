import numpy as np
import pandas as pd

import torch
from ignite.metrics import Accuracy
from .test import test


def _readout(model, loader, device='cpu'):
    with torch.no_grad():
        test_score = test(model, loader, loss_fn, device)

        trained_layer = model.linear
        
        out_features, in_features = model.linear.weight.shape        
        readout_layer = torch.nn.Linear(in_features, out_features)

        readout_layer.bias.fill_(0)
        readout_layer.weight.fill_(1)
        readout_layer.weight *= trained_layer.weight.sign()

        model.linear = readout_layer

        readout_score = test(model, loader, metrics, device)

        model.linear = trained_layer

    return test_score, readout_score


def readout_classifier_scores(models, data_loader, 
                                model_names=None, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_names is None:
        model_names = [m.type for m in models]

    scores, loss = [], Accuracy()
    for m in models:
        d = _readout(m, data_loader, {'acc', loss})
        scores.extend(s['acc'] for s in d)

    scores = pd.Series(
        data=scores,
        index=pd.MultiIndex.from_product(
            [model_names, ['standard', 'unit readout']], 
            names=['model', 'projection']
        ),
        name='accuracy'
    )

    return scores
