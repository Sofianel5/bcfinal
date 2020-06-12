from activation_functions.softmax import Softmax
from activation_functions.relu import ReLU
from activation_functions.tanh import Tanh
from operations.add import Add
import numpy as np
from operations.multiply import Multiply 

class Model():
    def __init__(self, layers_dim):
        self.bias = []
        self.weights = []
        for i in range(len(layers_dim)-1):
            self.weights.append(np.random.randn(layers_dim[i], layers_dim[i+1]) / np.sqrt(layers_dim[i]))
            self.bias.append(np.random.randn(layers_dim[i+1]).reshape(1, layers_dim[i+1]))

    
    def loss(self, input, label):
        mul = Multiply()
        add = Add()
        layer = ReLU()
        softmax = Softmax()
        for i in range(len(self.weights)):
            m = mul.forward(self.weights[i], input)
            b = add.forward(m, self.bias[i])
            input = layer.forward(b)
        return softmax.loss(input, label)
    
    def predict(self, input):
        mul = Multiply()
        add = Add()
        layer = ReLU()
        softmax = Softmax()
        for i in range(len(self.weights)):
            m = mul.forward(self.weights[i], input)
            b = add.forward(m, self.bias[i])
            input = layer.forward(b)
        p = softmax.predict(input)
        return np.argmax(p, axis=1)
    
    def train(self, input, label, num_epochs=100000, epsilon=0.01, reg_lambda=0.01):
        mulGate = Multiply()
        addGate = Add()
        layer = ReLU()
        softmaxOutput = Softmax()

        for epoch in range(num_epochs):
            # Forward propagation
            forward = [(None, None, input)]
            for i in range(len(self.weights)):
                mul = mulGate.forward(self.weights[i], input)
                add = addGate.forward(mul, self.bias[i])
                input = layer.forward(add)
                forward.append((mul, add, input))

            # Back propagation
            dtanh = softmaxOutput.diff(forward[len(forward)-1][2], label)
            for i in range(len(forward)-1, 0, -1):
                dadd = layer.backward(forward[i][1], dtanh)
                db, dmul = addGate.backward(forward[i][0], self.bias[i-1], dadd)
                dW, dtanh = mulGate.backward(self.weights[i-1], forward[i-1][2], dmul)
                dW += reg_lambda * self.weights[i-1]
                self.bias[i-1] += -epsilon * db
                self.weights[i-1] += -epsilon * dW
            if epoch % 1000 == 0:
                print("Loss after iteration %i: %f" %(epoch, self.loss(input, label)))
