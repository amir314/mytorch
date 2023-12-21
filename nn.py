from tensor import Tensor 

from layer import *


class Module:
    def __init__(self):
        self.layers:list[Layer]

    def forward(self, inputs:Tensor):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
    def get_grads(self):
        grads = []
        for layer in self.layers:
            grads.append(layer.grads)
        return grads