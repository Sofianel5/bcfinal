from activation_functions.softmax import Softmax
from activation_functions.relu import ReLU
from operations.add import Add
from operations.multiply import Multiply 

class Model():
    def __init__(self, layers_dim):
        self.bias = []
        self.weights = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers_dim[i], layers_dim[i+1]) / np.sqrt(layers_dim[i]))
            self.bias.append(np.random.randn(layers_dim[i+1]).reshape(1, layers_dim[i+1]))

    
    def loss(self, input, label):
        mul = Multiply()
        add = Add()
        layer = ReLU()
        softmax = softmax()
        for i in range(len(self.weights)):
            m = mul.forward(self.weights[i], input)
            b = mul.forward(m, self.bias[i])
            input = layer.forward(b)
        return softmax.loss(input, label)
    
    def predict(self, input):
        mul = Multiply()
        add = Add()
        layer = ReLU()
        softmax = softmax()
        for i in range(len(self.weights)):
            m = mul.forward(self.weights[i], input)
            b = mul.forward(m, self.bias[i])
            input = layer.forward(b)
        p = softmax.predict(input)
        return np.argmax(p, axis=1)
    
    def train(self, input, label, num_epochs=20000, epsilon=0.01, reg_lambda=0.01):
        mul = Multiply()
        add = Add()
        layer = ReLU()
        softmax = softmax()
        for epoch in range(num_epochs)