import numpy as np 
import nn 
from loss import MSE 
from optim import SGD

"""
We will approximate the function y = 2x from noisy data with a neural network. 
"""

np.random.seed(42)

# Data set 
n = 100
X_train = np.linspace(-100, 100, n).reshape(n,1)
y_train = 2*X_train + np.random.randn(n,1)

# Model
class MyNet(nn.Module):
    def __init__(self):
        self.layers = [nn.Linear(1,1)]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
model = MyNet()

# Loss 
loss_fn = MSE()

# Optimizer
lr = 0.01
optim = SGD(lr=lr, model=model)

# Training loop.
n_epoch = 5

for epoch in range(n_epoch):

    # Model predictions 
    preds = model.forward(X_train)
    
    # Loss and its gradient 
    loss = loss_fn.loss(predicted=preds, target=y_train)
    grad = loss_fn.grad(predicted=preds, target=y_train)
    
    # Backpropagation
    model.backward(grad=grad)

    # Optimize
    optim.step()

    print(f"Epoch: {epoch+1}")
    print(f"Training loss: {loss:.2f}\n")


print('The model reconstructed the following function:')
print(f'f(x) = {model.layers[0].params["w"].item()}*x + {model.layers[0].params["b"].item()}')