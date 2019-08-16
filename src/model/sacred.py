from sacred import Ingredient
from .init import init_model

model = Ingredient('model')
init_model = model.capture(init_model)
