import numpy as np
from operation import Operation 

class Multiply(Operation):
    def forward(self, weights, input):
        return np.dot(input, weights)
    
    def backward(self, weights, input, delta):
        dW = np.dot(np.transpose(input), delta)
        dX = np.dot(delta, np.transpose(weights))
        return dW, dX 
