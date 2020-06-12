# Liam Kronman and Sofiane Larbi's Mathematical Project
Creating an **Artificial Neural Network (ANN)** through the lens of calculus and precalculus.

## How to Run Our Project
First, make sure you have `python3` installed on your computer. It can be downloaded [here](https://www.python.org).  

Then type this line of code into your terminal, in the same directory that you have cloned this repository, to download necessary dataset packages:  

`pip install -r requirements.txt`  

Finally, run our primary python file by typing this line into your terminal:

`python3 mnist_nn.py`  


## What is an Artificial Neural Network (ANN)?
Imagine you have a bunch of images of dogs and cats, and you want to develop a program that will determine which images are of dogs and which are of cats. This may be harder than you think. Comparing the corresponding pixels of different images to a sort of "platonic ideal" of a dog or a cat is impractical, not only because the images might be of different dimensions, but also because the pictured pet must be in the *exact* same position as your model cat/dog, or you will probably get an incorrect output. For problems like these, that require independently finding connections in data, building an **ANN** may be the most effective solution.  

![Neural Network](/static/neuralnetwork.png)

An **ANN** is a series of layers made up of weighted nodes (called **neurons**) that feed into one another. The process begins with the **input layer**, a layer of fixed dimensionality matching the shape of the input matrix. As an input is passed through the first layer to the layers that are not seen to the user (known as **hidden layers**), the various neurons are "activated" to alter the values of the input. An output for each neuron is calculated by matrix multiplying the input by the weight matrix, and adding the bias vector. This output is fed into the neuron's activation function, which determines the final output for each neuron.  

To train our network, an algorithm called **gradient descent** is applied to the network. Since we know the output of the neural network, we are able to calculate the "loss" of the output, a score representing how wrong the neural network was in its output. For each node in the final layer, we are, in turn, able to calculate the derivative of the loss function with respect to each of its tunable parameters (the **weights** and the **bias**). We are able to nudge these parameters in the direction in which the derivative of loss with respect to parameter is most negative, thus decreasing the loss of the network. The output of the last layer is the returned output of the network, which should hopefully be the category the input fits best into.

## How Does Our ANN Work?
`mnist_nn.py` evaluates images of handwritten numbers and returns which numbers they represent. Please check out the **Glossary of Functions** below for more info on the referenced functions.  

Each image can be interpreted as a 28x28 vector of brightness values. The image is "flattened" into a one-dimensional 784-index-long vector, which is fed in as the **input layer**. Our network has an input layer of 784 neurons, two consecutive **hidden layers** of 100 neurons, and an output layer with 10 neurons. We also decided on a that our **batches** would have sizes of 50. A batch is a small subset of data that is added to the subset of data visible to the network that the network must create approximations/connections. First, you introduce less data elements to the network, so it isn't overwhelmed and is able to create approximations. As more batches are added to the visible dataset, the network can improve such approximations. Using batches rather than feeding each data point separately allows the network to better generalize its approximations since adding 50 samples reveals more about the complete dataset than adding one. We initialize the **gradients** (which will be used as a "map" to the most optimal approximation) of each layer as `d1prev`, `d2prev`, `d3prev`, which are vectors filled with 0s with the same shape as the weight matrix. When we begin a **training iteration** (one cycle of going forward in a network and backwards), we start by taking our label (the number 0-9 that the image is of) and creating a one-hot representation of it using `encode_one_hot`. We feed the image into the network in the `forward` function, where we reshape it into a 784-pixel-long flat image, add a bias, and go through the network by matrix multiplying the input by the bias, running the output through the activation function (sigmoid in this case), and adding a bias.  

The network is initialized by creating the bias matrix in `add_bias` and the weight matrix in `init_weight`. These matrices correspond to the the parameters on all the nodes in each layer. Each node in the input layer is connected to every node in the first hidden layer. The input values are multiplied by a weight matrix and a bias term is added. This process continues until the final layer of the network. In order to increase the accuracy of the network, we calculate a 'loss function' which returns a value representing how wrong the network's prediction is. We find the derivative (**gradient**) of the loss function and modify the parameters of the network, so as to travel down the gradient (**gradient descent**). This increases the accuracy of the network over multiple iterations.


## What Calculus Principles are Involved in This Project?

## What Precalculus Principles are Involved in This Project?

## Glossary of Functions
### In mnist_nn.py (handwriting recognition)
`encode_one_hot`:  Converts a value of enumerated type into a one-hot encoding. For instance the number 3, if only the values from 0-9 are possible would become [0,0,0,1,0,0,0,0,0,0] after being run through this function.

`add_bias`: Adds a bias term. Similar to the +b in the slope-intercept form for a line (y=mx+b). Depending on the parameter, this function appends a bias vector as either a row or a column to an input matrix.

`init_weight`: Initializes a weight matrix. Returns a tuple of the weight matrix for layers 1, 2, and 3. The weights are initialized randomly.

`forward`: Computes an output vector for a layer of the network. For each layer, an output is computed by multiplying the input vector by the weight matrix and adding the bias vector, both of which are stored in each node. The output vector is a one dimensional vector of size 1x10, and each index corresponds to the networks's belief that the image shown was that number.

`predict`: Computes the final output by finding the maximum probability in the output vector.

`compute_loss`: Computes the log loss of the output value using categorical cross entropy. We use this rather than binary assessments of "correct" or "incorrect" because gradient descent requires a differentiable function to compute and travel down the gradient.

`backward`: Computes the gradient of the network using backpropagation. Then, calculates the derivative of the loss function with respect to each parameter in the last layer. To find the partial derivatives in the further layers of the network (the layers closer to input), we use the chain rule and multiply the derivatives.

`gen_data`: Downloads and shapes the data from MNIST, a popular dataset of handwriting images. Each image is a 28x28 matrix of pixel brightness values. The image is "flattened" into a one dimensional vector of size 784 when fed into the network.

### OOP method
`ActivationFunction`: `forward` defines the output of the function, while `backward` defines the derivative multiplied by another differential.

`ReLU`: Short for Rectified Linear Unit. An almost-linear activation function defined by the piecewise function 0 if x < 0 else x. The nonlinearity of this function allows the neural network to approximate nonlinear functions. Since this function is so computationally efficient, it is essentially standard across all kinds of neural networks.

`Sigmoid`: Another activation function defined by 1/(1+e^-x). This function is also popular, but not as useful as ReLU since it is less computationally efficient.

`Tanh`: Hyperbolic tangent function.

`Add`: The vector addition operation.

`Multiply`: The vector multiplication operation (dot product).

`Model`:
