"""
An optimizer will be used to minimize our loss by adjusting the 
parameters of a model. 
"""

from nn import Module

import numpy as np


class Optimizer:
    def __init__(self, lr: float, model:Module):
        self.lr = lr
        self.model = model 

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float, model:Module):
        super().__init__(lr, model)

    def step(self):
        for layer in self.model.layers:
            for key in layer.params.keys():
                layer.params[key] -= np.multiply(self.lr, layer.grads[key])