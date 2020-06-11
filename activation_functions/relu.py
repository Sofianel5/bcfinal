import numpy as np 
from activation_function import ActivationFunction

class ReLU(ActivationFunction):
    
    def forward(self, inputs):
        return np.max(0,inputs)

    def backward(self, inputs, differential):
        return 0 if inputs > 0 else 1 