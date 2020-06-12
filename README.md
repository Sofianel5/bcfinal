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
Our program evaluates images of handwritten numbers and returns which numbers they represent. Please check out the **Glossary of Functions** below for more info on the referenced functions.  


## What Calculus Principles are Involved in This Project?

## What Precalculus Principles are Involved in This Project?

## What precalculus principles are involved in this project?

## Glossary of Functions
### In mnist_nn.py (handwriting recognition)
`encode_one_hot`:  Converts an enum into a one hot encoding. For instance the number 3, if only 1-10 is possible would become [0,0,1,0,0,0,0,0,0,0]

`add_bias`: Adds a bias term. Similar to the +b in the equation for a line. Depending on the parameter, it appends a bias vector as either a row or a column to an input matrix.

`init_weight`: Initializes a weight matrix. Returns a touple of the weight matrix for layers 1, 2, and 3. The weights are initialized randomly.

`forward`: Computes an output vector for the network. For each layer, an output is computed by multiplying the input vector by the weight matrix and adding the bias vector. The output vector a one dimentional vector of shape (10,) and each index corresponds to the networks's belief that the image shown was that number.

`predict`: Computes the final output by finding the maximum probability in the output vector.

`compute_loss`: Computes the log loss of the output value using catagorical cross entropy. We use this rather than binary rewards of correct or incorrect because gradient descent requires a differentiable function in order to compute and travel down the gradient. 

`backward`: Computes the gradient of the network using backpropogation. The derivative of the loss function with respect to each parameter in the last layer is calculated. In order to find the partial derivatives in the further (closer to input) layers of the network, we use the chain rule and multiply the derivatives. 

`gen_data`: Downloads and shapes the data from MNIST, a popular dataset of handwriting. Each image is a 28*28 matrix of pixel brightness values. The image is 'flattened' into a one dimentional vector of size 784 when fed into the network. 
