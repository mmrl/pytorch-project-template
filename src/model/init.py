import os
import yaml
import torch
import torch.nn as nn
import torch.nn.init as init


###############################################################################
# RNN WRAPPERS
###############################################################################

class RNNPredictor(nn.Module):
    def __init__(self, rnn, output_size, predict_last=True):
        super(RNNPredictor, self).__init__()
        self.rnn = rnn
        self.linear = nn.Linear(rnn.hidden_size, output_size)
        self.predict_last = predict_last

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.linear.reset_parameters()

    @property
    def input_size(self):
        return self.rnn.input_size

    @property
    def output_size(self):
        return self.linear.weight.shape[1]

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

    @property
    def n_layers(self):
        return self.rnn.num_layers

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)

        if self.predict_last:
            pred = self.linear(output[:, -1, :])
        else:
            tsteps = self.input.size(1)
            pred = [self.linear(output[:, i, :]) for i in range(tsteps)]
            pred = torch.stack(pred, dim=1)

        return pred, hidden


###############################################################################
# Model Initialization
###############################################################################


def _weight_init_(module, init_fn_):
    try:
        init_fn(m.weight.data)
    except AttributeError:
        for layer in self.all_weights:
            w, r = layer[:2]
            init_fn_(w)
            init_fn_(r)


def weight_init_(rnn, mode=None, **kwargs):
    if mode == 'xavier':
        _weight_init_(rnn, lambda w: init.xavier_uniform_(w, **kwargs))
    elif mode == 'orthogonal':
        _weight_init_(rnn, lambda w: init.orthogonal_(w, **kwargs))
    elif mode == 'kaiming':
        _weight_init_(rnn, lambda w: init.kaiming_uniform_(w, **kwargs))
    elif mode != None:
        raise ValueError(
                'Unrecognised weight initialisation method {}'.format(mode))


def init_model(model_type, hidden_size, input_size, n_layers,
        output_size, dropout=0.0, weight_init=None, device='cpu',
        predict_last=True
    ):
    if model_type == 'RNN':
        rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    elif model_type == 'LSTM':
        rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    elif model_type == 'GRU':
        rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    else:
        raise ValueError('Unrecognized RNN type')

    weight_init_(rnn, weight_init)

    model = RNNPredictor(
        rnn=rnn,
        output_size = output_size,
        predict_last=predict_last
    ).to(device=device)

    return model


###############################################################################
#  Load models
###############################################################################


def load_meta(path):
    with open(path, mode='r') as f:
        meta = yaml.safe_load(f)
    return meta


def _load_model(meta, model_file):
    meta = load_meta(meta)
    with open(model_file, mode='rb') as f:
        state_dict = torch.load(f)
        if 'model-state' in state_dict:
            state_dict = state_dict['model-state']
    m = init_model(device='cpu', **meta['model-params'])
    m.load_state_dict(state_dict)

    return m


def load_model(model_folder):
    meta = os.path.join(model_folder, 'meta.yaml')
    for file in os.listdir(model_folder):
        if file.endswith((".pt", ".pth")):
            file = os.path.join(model_folder, file)
            return _load_model(meta, file)
