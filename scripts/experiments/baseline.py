import sys

import torch
import ignite
from ignite.engine import Events
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver()

sys.path.insert(0, '../../src')

# Load experiment ingredients and their respective configs.
from dataset.sacred import dataset, load_delayed_addition
from model.sacred import model, init_model
from training.handlers import Tracer
from training.sacred import training, set_seed_and_device, \
                            setup_training, run_training

dataset.add_config('configs/dummy-dataset.yaml')
training.add_config('configs/dummy-training.yaml')

# Configs for both models. Should execute using 'with model.<config>'
model.add_named_config('lstm', 'configs/dummy-lstm.yaml')

# Create experiment
ex = Experiment(
    name='Dummy Experiment',
    ingredients=[dataset, model, training]
)

# Runtime options
save_folder = '../../data/sims/test/'
ex.add_config({
    'save': save_folder,
    'no_cuda': False,
})

# Add dependencies
ex.add_package_dependency('torch', torch.__version__)

# Add observer
ex.observers.append(
    FileStorageObserver.create(save_folder))
# ex.observers.append(
#     MongoObserver.create(url='127.0.0.1:27017',
#                         db_name='MY_DB')
# )

@ex.capture
def log_training(tracer):
    ex.log_scalar('training_loss', tracer.trace[-1])
    tracer.trace.clear()

@ex.capture
def log_validation(engine):
    for metric, value in engine.state.metrics.items():
        ex.log_scalar('val_{}'.format(metric), value)


@ex.automain
def main(_config, seed):
    save = _config['save']
    no_cuda = _config['no_cuda']
    batch_size = _config['training']['batch_size']

    device = set_seed_and_device(seed, no_cuda)
    training_set, test_set, validation_set = load_delayed_addition(
                                                batch_size=batch_size)
    model = init_model(device=device)

    trainer, validator, checkpoint, metrics = setup_training(
        model, validation_set,
        save=save, device=device,
        trace=False, time=False)[:4]

    tracer = Tracer().attach(trainer)
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, lambda e: log_training(tracer))
    validator.add_event_handler(Events.EPOCH_COMPLETED, log_validation)

    test_metrics = run_training(
        model=model,
        train_data=training_set,
        trainer=trainer,
        test_data=test_set,
        metrics=metrics,
        model_checkpoint=checkpoint,
        device=device
    )

    # save best model performance and state
    for metric, value in test_metrics.items():
        ex.log_scalar('test_{}'.format(metric), value)

    ex.add_artifact(str(checkpoint._saved[-1][1][0]), 'trained-model')
