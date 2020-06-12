import numpy as np
from .operation import Operation 

class Multiply(Operation):
    def forward(self, weights, input):
        return np.dot(input, weights)
    
    def backward(self, W, X, dZ):
        dW = np.dot(np.transpose(X), dZ)
        dX = np.dot(dZ, np.transpose(W))
        return dW, dX
