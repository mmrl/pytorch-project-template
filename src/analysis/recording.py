import os
import numpy as np
import pandas as pd
import torch


def record_activations(model, data, device):
    raw_data = []
    model.to(device)

    # set up recording forward hook
    def acitvation_recorder(self, input, output):
        out, _ = output
        try:
            out = out.numpy()
        except TypeError:
            out = out.cpu().numpy()
        raw_data.append(out)

    hook = model.rnn.register_forward_hook(acitvation_recorder)

    # feed stimuli to network
    with torch.no_grad():
        for i, batch in enumerate(data):
            inputs, _ = batch
            inputs = inputs.to(device)

            outputs = model(inputs)[0]

    hook.remove()
    raw_data = np.concatenate(raw_data)

    # Transform data to Pandas DataFrame

    input_idx = range(raw_data.shape[0])
    timesteps = range(raw_data.shape[1])
    units =  range(raw_data.shape[2])

    s = pd.Series(
        data=raw_data.reshape(-1),
        index=pd.MultiIndex.from_product(
            [input_idx, timesteps, units],
            names=['input','timestep', 'unit']),
        name='activation')

    return s


def record_from_models(models, data, device):
    recordings = [
        record_activations(model, test_data, device) for model in models
    ]
    return recordings


# def weighted_activity(model, recordings):
#     df = recordings.groupby(['input', 'unit']).last()
#     weights = model.linear.weight.detach().numpy().T.reshape(-1)

#     gb_act = df.set_index('class', append=True).groupby(['unit', 'class'])

#     def weigh(group):
#         # name in the same order given in groupby
#         unit, label = group.name
#         return group * weights[label, unit]

#     weighted_activity = gb_act.apply(weigh)

#     return weighted_activity


# def mean_weighted_activity(model, recordings):
#     df = recordings.groupby(['input', 'unit']).last()
#     df = df.set_index(
#         'class', append=True).groupby(['unit','class']).mean()
#     df.columns = ['mean activation']

#     df['weight'] = weights.T.reshape(-1)

#     wact = weighted_activity(recordings, model)

#     df['mean weighted activation'] = wact.groupby(['unit', 'class']).mean()

#     return df
