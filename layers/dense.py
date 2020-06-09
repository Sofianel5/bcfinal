import numpy as np
np.random.seed(0)

X = [
    [1.0,2.0,3.0,2.5],
    [2.0,5.0,1.0,2.0],
    [-1.5,2.7,3.3,-0.8]
]

class DenseLayer():
    def __init__(self, inputshape, numneurons):
        self.weights = 0.10 * np.random.randn(inputshape, numneurons)
        self.biases = np.zeros((1, numneurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


