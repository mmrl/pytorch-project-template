import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Events
from ignite.metrics import Loss, Accuracy

from .engine import create_rnn_trainer, create_rnn_evaluator
from .handlers import Tracer, ModelCheckpoint, EarlyStopping, \
    Timer, LRScheduler, ProgressLog
from .optimizer import init_optimizer, init_lr_scheduler
from .loss import RNNLossWrapper, RateDecay, ComposedLoss


def set_seed_and_device(seed, no_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not no_cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def setup_training(
        model, validation_data, optim, lr, l2_norm, rate_reg,
        clip, early_stopping,  decay_lr, lr_scale, lr_decay_patience,
        keep_hidden, save, device, trace=False, time=False
    ):

    criterion = nn.MSELoss()
    metrics = {'mse': Loss(criterion)}
    loss = RNNLossWrapper(criterion)
    if rate_reg > 0:
        loss = ComposedLoss(
            terms=[loss, RateDacay(device=device)],
            decays=[1.0, rate_reg]
        )

    optimizer = init_optimizer(
        optim, model.parameters(), lr, l2_norm)

    trainer = create_rnn_trainer(
        model, optimizer, loss,
        grad_clip=clip, device=device, reset_hidden=(not keep_hidden)
    )

    validator = create_rnn_evaluator(model, metrics, device=device)
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(validation_data)

    # Tracing
    if trace:
        training_tracer = Tracer().attach(trainer)
        validation_tracer = Tracer(metrics.keys()).attach(validator)
    else:
        training_tracer, validation_tracer = None, None

    # Training time
    if time:
        timer = Timer(average=False)
        timer.attach(trainer)
    else:
        timer = None

    # Add handlers. Learning rate decay
    if decay_lr:
        lr_scheduler = hdlr.LRScheduler(
            loss='mse', scheduler=init_lr_scheduler(
                                        optimizer, 'reduce-on-plateau',
                                        lr_decay=lr_scale,
                                        patience=lr_decay_patience))
        lr_scheduler.attach(validator)

    # Model checkpoint and early stopping
    def score_fn(engine):
        return -engine.state.metrics['mse']

    checkpoint = ModelCheckpoint(
        dirname=save,
        filename_prefix='',
        score_function=score_fn,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True
    )
    validator.add_event_handler(
        Events.COMPLETED, checkpoint, {'model': model})

    if early_stopping:
        stopper = EarlyStopping(
            patience=100,
            score_function=score_fn,
            trainer=trainer
        )
        validator.add_event_handler(Events.COMPLETED, stopper)

    return trainer, validator, checkpoint, metrics, \
        training_tracer, validation_tracer, timer


def setup_logging(trainer, validator, metrics, n_batches, log_interval):
    logger = ProgressLog(n_batches, log_interval)
    logger.attach(trainer, validator, metrics.keys())
