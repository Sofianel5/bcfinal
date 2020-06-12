import numpy as np 
from .activation_function import ActivationFunction

class Tanh(ActivationFunction):
    
    def forward(self, inputs):
        return np.tanh(inputs)

    def backward(self, inputs, differential):
        output = self.forward(inputs)
        return (1.0 - np.square(output)) * differential