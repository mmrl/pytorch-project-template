from sacred import Ingredient
from .setup import setup_training, set_seed_and_device
from .engine import run_training

training = Ingredient('training')
set_seed_and_device = training.capture(set_seed_and_device)
setup_training = training.capture(setup_training)
run_training = training.capture(run_training)
