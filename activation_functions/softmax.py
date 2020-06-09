import numpy as np 
from activation_function import ActivationFunction

class Softmax(ActivationFunction):
    def forward(self, inputs):
        ex = np.exp(inputs-np.max(inputs))
        return ex/ex.sum()
    def backward(self, inputs, differential):
        pass