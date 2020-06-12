import numpy as np 
from .activation_function import ActivationFunction

class ReLU(ActivationFunction):

    def forward(self, inputs):
        return np.maximum(inputs, 0, inputs)

    def backward(self, inputs, differential):
        return np.greater(inputs, 0).astype(int) * differential