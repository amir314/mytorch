"""
A neural network is made up of layers, which are just functions. For example, when given an 
input x, a linear layer computes w @ x + b, where w is a matrix and b a vector of appropriate 
dimensions.
"""

from tensor import Tensor 

import numpy as np

from typing import Dict


class Layer:
    """
    A layer is simply a function. A neural network is a composition of several layers: f_1(f_2(...(f_L))).
    """
    
    def __init__(self, params: Dict[str, Tensor]): 
        self._params: Dict[str, Tensor] = params
        self.grads: Dict[str, Tensor] = {}

    @property 
    def params(self):
        return self._params
    
    @params.setter 
    def params(self, new_params:Tensor):
        self._params = new_params

    def forward(self, inputs: Tensor) -> Tensor:
        """
        This method defines how the layer transforms its inputs. 
        The inputs are expected to have shape (batch_size, input_length).
        """

        raise NotImplementedError
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        This method defines how the layer passes its gradients backwards.
        """

        raise NotImplementedError
    

class Linear(Layer):
    def __init__(self, in_features:int, out_features:int):
        """
        Initializes linear layer. 
        
        in_features: Length of inputs in batch. 
        out_featurs: Desired output length.  
        """

        self.in_features = in_features
        self.out_features = out_features

        params = {}
        params['w'] = np.random.randn(in_features, out_features)
        params['b'] = np.random.randn(out_features)

        super().__init__(params)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Expects an input of size (batch_size, in_features).
        Calculates x @ w + b for every input x in the batch. 

        inputs: Tensor of size (batch_size, in_features).
        Returns: Tensor of size (batch_size, out_features).
        """

        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        Calculates gradients of the loss wrt. the parameters of this layer. Passes 
        on the gradients of the loss wrt. the outputs of this layer. 

        grad: Tensor of size (batch_size, out_features)
        Returns: Tensor of size (batch_size, in_features)
        """

        n = grad.shape[0]
        self.grads['w'] = np.divide((self.inputs.T @ grad), n)
        self.grads['b'] = np.mean(grad, axis=0)
        return grad @ self.params['w'].T
    

class ReLU(Layer):
    def __init__(self):
        super().__init__(params={})

    def forward(self, inputs:Tensor):
        self.inputs = inputs
        return inputs*(inputs>0)
    
    def backward(self, grad):
        return grad*(self.inputs>0)