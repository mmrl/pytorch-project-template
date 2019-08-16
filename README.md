# PyTorch Project Template

This is a project template for [PyTorch](https://pytorch.org/) using [Ignite](https://github.com/pytorch/ignite) and [Sacred](https://github.com/IDSIA/sacred) to run, store and keep track off all the experiments. Ignite provides the different engines and event handlers to train and track the progress of the runs. Sacred on the other hand provides functionality to define and store results in a way that it is easy to reproduce and retrieve them for analysis.

## Dependencies

This is the list of main libraries:

* [PyTorch and Torchvision](https://pytorch.org/): Basic framework for Deep Learning models and training.
* [Ignite](https://github.com/pytorch/ignite): High-level framework to train models, eliminating the need for much boilerplate code.
* [Sacred](https://github.com/IDSIA/sacred): Libary used to define and run experiments in a systematic way.
* [Jupyter](https://jupyter.org/): The default way to visualise and analise results.

Optional libraries (but highly recommended):

* [GitPython](https://github.com/gitpython-developers/GitPython): Useful to integrate Sacred to Git. This allows to keep track of the status of the repo when an experiment was ran.
* [MongoDB](https://www.mongodb.com/): To store the results in database instead of simple files.
* [PyMongo](https://api.mongodb.com/python/current/): Needed if you want to use MongoDB.
* [Incense](https://github.com/JarnoRFB/incense): To load experiments from a MongoDB database into a Jupyter Notebook.
* [Omniboard](https://github.com/vivekratnavel/omniboard): To visualise the results contained in a database.

## Instalation

Refer to each page for how to install the corresponding libraries and their dependencies. Then, we can create a project from the template using GitHub's template functionality. For more details refer to ["Creating a repository project board"](https://help.github.com/en/articles/creating-a-repository-from-a-template) on GitHub.
