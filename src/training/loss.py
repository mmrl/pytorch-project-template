import torch
import torch.nn.modules.loss as L
import torch.nn.functional as F
import ignite.metrics as metrics


class RNNLossWrapper(L._Loss):
    def __init__(self, loss_fn, reduction='mean'):
        super(RNNLossWrapper, self).__init__(reduction=reduction)
        self._loss_fn = loss_fn

    def forward(self, input, target):
        pred, _ = input
        return self._loss_fn(pred, target)


class RateDecay(L._Loss):
    def __init__(self, target=0, reduction='mean', device='cpu'):
        super(RNNHiddenL1Loss, self).__init__(reduction)

        self.target = torch.as_tensor(target).to(device)

    def forward(self, input, target):
        _, hidden = input

        return F.l1_loss(hidden, self.target, reduction=self.reduction)


class L1WeightDecay(L._Loss):
    def __init__(self, model_parameters, target=0, device='cpu'):
        super(L1WeightDecay, self).__init__('none')
        self.parameters = model_parameters
        self.target = torch.as_tensor(target).to(device)

    def forward(self, input, target):
        return F.l1_loss(self.parameters, self.target, reduction='none')


class ComposedLoss(L._Loss):
    def __init__(self, terms, weights):
        super(ComposedLoss, self).__init__(reduction='none')

        self.terms = terms
        self.weights = weights

    def forward(self, input, target):
        value = 0
        for term, w in zip(self.terms, self.weights):
            value += w * term(input, target)

        return value
