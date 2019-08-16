from sacred import Ingredient
from .dummy import load_data

dataset = Ingredient('dataset')
load_delayed_addition = dataset.capture(load_data)
