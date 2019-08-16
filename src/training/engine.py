import numpy as np
import csv

import torch
import torch.nn as nn

from ignite.engine import Events, Engine, _prepare_batch
from tqdm import tqdm

########################################################################################
# Training
########################################################################################


def _detach_hidden_state(hidden_state):
    """
    Use this method to detach the hidden state from the previous batch's history.
    This way we can carry hidden states values across training which improves
    convergence  while avoiding multiple initializations and autograd computations
    all the way back to the start of start of training.
    """

    if hidden_state is None:
        return None
    elif isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    elif isinstance(hidden_state, list):
        return [_detach_hidden_state(h) for h in hidden_state]
    elif isinstance(hidden_state, tuple):
        return tuple(_detach_hidden_state(h) for h in hidden_state)
    raise ValueError('Unrecognized hidden state type {}'.format(type(hidden_state)))


def create_rnn_trainer(model, optimizer, loss_fn, grad_clip=0, reset_hidden=True,
                    device=None, non_blocking=False, prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _training_loop(engine, batch):
        # Set model to training and zero the gradients
        model.train()
        optimizer.zero_grad()

        # Load the batches
        inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
        hidden = engine.state.hidden

        # Forward pass
        pred, hidden = model(inputs, hidden)
        loss = loss_fn((pred, hidden), targets)

        # Backwards
        loss.backward()

        # Optimize
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if not reset_hidden:
            engine.state.hidden = hidden

        return loss.item()

    # If reusing hidden states, detach them from the computation graph
    # of the previous batch. Usin the previous value may speed up training
    # but detaching is needed to avoid backprogating to the start of training.
    def _detach_wrapper(engine):
        if not reset_hidden:
            engine.state.hidden = _detach_hidden_state(engine.state.hidden)

    engine = Engine(_training_loop)
    engine.add_event_handler(Events.EPOCH_STARTED, lambda e: setattr(e.state, 'hidden', None))
    engine.add_event_handler(Events.ITERATION_STARTED, _detach_wrapper)

    return engine


def create_rnn_evaluator(model, metrics, device=None, hidden=None, non_blocking=False,
                        prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
            pred, _ = model(inputs, hidden)

            return pred, targets

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def run_training(
        model, train_data, trainer, epochs,
        metrics, test_data, model_checkpoint, device
    ):
    trainer.run(train_data, max_epochs=epochs)

    # Select best model
    best_model_path = str(model_checkpoint._saved[-1][1][0])
    with open(best_model_path, mode='rb') as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)

    tester = create_rnn_evaluator(model, metrics, device=device)
    tester.run(test_data)

    return tester.state.metrics
