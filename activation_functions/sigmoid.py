import numpy as np 
from activation_function import ActivationFunction

class Sigmoid(ActivationFunction):
    
    def forward(self, inputs):
        return 1.0 / (1.0 + np.exp(-inputs))

    def backward(self, inputs, differential):
        output = self.forward(inputs)
        return (1.0 - output) * output * differential