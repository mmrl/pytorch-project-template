
import sys
import os
import argparse
import datetime
import yaml

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Events
from ignite.metrics import Loss, Accuracy

sys.path.insert(0, '../../src')

from dataset.dummy import load_data
from model.init import init_model
from training.setup import set_seed_and_device, setup_training, setup_logging
from training.engine import run_training


def train_model(
    # Model parameters
    model='subLSTM', nlayers=1, nhid=50, dropout=0.0,
    # Data paramters here
    # Training parameters
    epochs=10, batch_size=50, optim='RMSprop', lr=1e-4, l2_norm=0.0, rate_reg=0.0,
    clip=1.0, early_stopping=False, decay_lr=False, lr_scale=0.1,
    lr_decay_patience=10, keep_hidden=False,
    # Replicability and storage
    save='../../data/sims/deladd/test', seed=18092, no_cuda=False,
    verbose=False, log_interval=10
):
    # Set training seed and get device
    device = set_seed_and_device(seed, no_cuda)

    # Load training data
    train_data, test_data, validation_data = load_data()

    # Initialise model
    input_size, hidden_size, n_responses = 2, nhid, 1
    model = init_model(
        model_type=model,
        n_layers=nlayers, hidden_size=nhid,
        input_size=input_size, output_size=n_responses,
        device=device,
        dropout=dropout
    )

    # Set up the training regime
    setup = setup_training(
        model, validation_data, optim, lr, l2_norm,
        rate_reg, clip, early_stopping,
        decay_lr, lr_scale, lr_decay_patience,
        keep_hidden, save, device, True, True,
    )

    trainer, validator, checkpoint, metrics = setup[:4]
    training_tracer, validation_tracer, timer = setup[4:]

    if verbose:
        setup_logging(
            trainer, validator, metrics,
            len(train_data), log_interval
        )

    # Run training
    test_metrics = run_training(
        model=model,
        train_data=train_data,
        trainer=trainer,
        epochs=epochs,
        test_data=test_data,
        metrics=metrics,
        model_checkpoint=checkpoint,
        device=device
    )

    # Testing preformance
    test_loss = test_metrics['mse']

    print('Training ended: test loss {:5.4f}'.format(
        test_loss))

    print('Saving results....')

    # Save traces
    training_tracer.save(save)
    validation_tracer.save(save)

    # Save experiment metadata
    model_params = {
        'model_type': model,
        'hidden_size': hidden_size,
        'n_layers': nlayers,
        'input_size': input_size,
        'output_size': n_responses,
        'dropout': dropout,
    }

    learning_params = {
        'optimizer': optim,
        'learning-rate': lr,
        'l2-norm': l2_norm,
        'criterion': 'mse',
        'batch_size': batch_size,
    }

    # Save data parameters in a dictionary for testing
    data_params = {
        'seqlen': seqlen, 'naddends': naddends,
        'minval': minval, 'maxval': maxval,
        'train-size': training_size, 'test-size': testing_size,
        'train-noise': train_noise, 'test-noise': test_noise,
        'val-split': val_split, 'keep-hidden': keep_hidden
    }

    meta = {
        'data-params': data_params,
        'model-params': model_params,
        'learning-params': learning_params,
        'info': {
            'test-score': test_loss,
            'training-time': timer.value(),
            'timestamp': datetime.datetime.now()
        },
        'seed': seed
    }

    with open(save + '/meta.yaml', mode='w') as f:
        yaml.dump(meta, f)

    print('Done.')


##############################################################################
# PARSE THE INPUT
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train an LSTM variant on the Delayed Addition task')

    # Model parameters
    parser.add_argument('--model', type=str, default='subLSTM',
                        help='RNN model tu use. One of:'
                        'subLSTM|fix-subLSTM|LSTM|GRU')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--nhid', type=int, default=50,
                        help='number of hidden units per layer')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='the drop rate for each layer of the network')

    # Data parameters
    parser.add_argument('--seqlen', type=int, default=50,
                        help='sequence length')
    parser.add_argument('--naddends', type=int, default=2,
                        help='the number of addends to be unmasked in a sequence'
                        'must be less than the sequence length')
    parser.add_argument('--minval', type=float, default=0.0,
                        help='minimum value of the addends')
    parser.add_argument('--maxval', type=float, default=1.0,
                        help='maximum value of the addends')
    parser.add_argument('--train-noise', type=float, default=0.0,
                        help='variance of the noise to add to the data')
    parser.add_argument('--test-noise', type=float, default=0.0,
                        help='variance of the noise to add to the data')
    parser.add_argument('--training-size', type=int, default=1000,
                        help='size of the randomly created training set')
    parser.add_argument('--testing-size', type=int, default=10000,
                        help='size of the randomly created test set')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='proportion of trainig data used for validation')
    parser.add_argument('--fixdata', action='store_true',
                        help='flag to keep the data fixed across each epoch')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=40,
                        help='max number of training epochs')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='size of each batch. keep in mind that if the '
                        'training size modulo this quantity is not zero, then'
                        'the it will be increased to create a full batch.')
    parser.add_argument('--optim', type=str, default='rmsprop',
                        help='gradient descent method, supports on of:'
                        'adam|sparseadam|adamax|rmsprop|sgd|adagrad|adadelta')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--l2-norm', type=float, default=0,
                        help='weight of L2 norm')
    parser.add_argument('--rate-reg', type=float, default=0.0,
                        help='regularization factor for hidden variables')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping')
    parser.add_argument('--early-stopping', action='store_true',
                        help='use early stopping')
    parser.add_argument('--decay_lr', action='store_true',
                        help='if provided will decay the learning after not'
                        'improving for a certain amount of epochs')
    parser.add_argument('--lr-scale', type=float, default=0.1,
                        help='the factor by which learning rate should be scaled'
                        'after the patience has been exhausted')
    parser.add_argument('--lr_decay_patience', type=int, default=10,
                        help='specifies the number of epochs to wait until'
                        'decay is applied before applying the decay factor')
    parser.add_argument('--keep-hidden', action='store_true',
                        help='keep the hidden state values across an epoch'
                        'of training, detaching them from the computation graph'
                        'after each batch for gradient consistency')

    # Replicability and storage
    parser.add_argument('--save', type=str,
                        default='../../data/sims/deladd/test',
                        help='path to save the final model')
    parser.add_argument('--seed', type=int, default=18092,
                        help='random seed')

    # CUDA
    parser.add_argument('--no-cuda', action='store_true',
                        help='flag to disable CUDA')

    # Print options
    parser.add_argument('--verbose', action='store_true',
                        help='print the progress of training to std output.')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='report interval')

    args = parser.parse_args()

    train_model(**vars(args))
