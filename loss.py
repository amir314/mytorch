"""
Ultimately, we want to use neural networks to make predictions. 
A loss function computes how close a prediction is to the actual value.
By minimizing the loss function (for which we need its gradients), we can optimize our network.
"""

from tensor import Tensor

import numpy as np


class Loss:
    def __init__(self):
        pass

    def loss(self, predicted:Tensor, target:Tensor):
        raise NotImplementedError
    
    def grad(self, predicted:Tensor):
        raise NotImplementedError


class MSE(Loss):
    def loss(self, predicted:Tensor, target:Tensor):
        return np.mean((predicted-target)**2) 
    
    def grad(self, predicted:Tensor, target:Tensor):
        n = predicted.shape[0]
        return np.multiply((predicted-target), 2/n)