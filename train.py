import matplotlib.pyplot as plt
import numpy as np 
from layers.model import Model
import random
import math
import sklearn
import sklearn.datasets
import sklearn.linear_model
random.seed(0)

""" 
Training the neural net to tell the difference between sinh(x) and its taylor series approximation,
x + x^3/3! + x^5/5!
0 = sinh
1 = taylor
"""
inputs = []
labels = []
SIZE = 200
for i in range(SIZE):
    if random.random() > 0.5:
        # add a sinh
        x = random.randrange(-4, 4)
        y = np.sinh(x)
        inputs.append(np.array([x, y], dtype=np.float64)) 
        labels.append(0)
    else:
        # add a taylor approximation
        x = random.randrange(-4, 4)
        y = x + x**3/math.factorial(3) + x**5/math.factorial(5)
        inputs.append(np.array([x,y ], dtype=np.float64))
        labels.append(1)

X = np.array(inputs)
y = np.array(labels)
#X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
layers_dim = [2, 3, 2]

model = Model(layers_dim)
model.train(X, y)


