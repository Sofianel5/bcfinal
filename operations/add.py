import numpy as np
from .operation import Operation 

class Add(Operation):
    def forward(self, input, bias):
        return input + bias
    
    def backward(self, input, bias, delta):
        dX = delta * np.ones_like(input)
        db = np.dot(np.ones((1, delta.shape[0]), dtype=np.float64), delta)
        return db, dX 
