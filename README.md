# Liam Kronman and Sofiane Larbi's Mathematical Project
Creating an **Artificial Neural Network (ANN)** through the lens of calculus and precalculus.

## How to Run Our Project
First, make sure you have `python3` installed on your computer. It can be downloaded [here](https://www.python.org).  

Then type this line of code into your terminal, in the same directory that you have cloned this repository, to download necessary dataset packages:  

`pip install -r requirements.txt`  

Finally, run our primary python file in your terminal with this line:

`python3 mnist_nn.py`  


## What is an Artificial Neural Network (ANN)?
Imagine you have a bunch of images of dogs and cats, and you want to develop a program that will determine which images are of dogs and which are of cats. This may be harder than you think. Comparing the corresponding pixels of different images to a sort of "platonic ideal" of a dog or a cat is impractical, not only because the images might be of different dimensions, but also because if the pictured pet is not in the *exact* same position as your model cat/dog, you will probably get an incorrect output. For problems like these, that require independently finding connections in data, building an **ANN** may be the most effective solution.  

![Neural Network](/static/neuralnetwork.png)

An **ANN** is a series of layers made up of weighted nodes (called **neurons**) that feed into one another. The process begins with the **input layer**, a layer of fixed dimensionality matching the shape of the input matrix. As an input is passed through the first layer to the layers that are not seen to the user (known as **hidden layers**), the various neurons are "activated" to alter the values of the input. An output for each neuron is calculated by matrix multiplying the input by the weight matrix, and adding the bias vector. This output is fed into the neuron's activation function, which determines the final output for each neuron.  

To train our network, an algorithm called **gradient descent** is applied to the network. Since we know the output of the neural network, we are able to calculate the "loss" of the output, a score representing how wrong the neural network was in its output. For each node in the final layer, we are able to calculate the derivative of the loss function with respect to each of its tunable parameters in turn (the **weights** and the **bias**). We are able to nudge these parameters in the direction in which the derivative of loss with respect to parameter is most negative, thus decreasing the loss of the network. The output of the last layer is the returned output of the network, which should hopefully be the category the input fits best into.

## How Does Our ANN Work?
Please check out the **Glossary of Functions** below for more info on the referenced functions. Our program evaluates images of handwritten numbers and returns which numbers they represent.  


## What Calculus Principles are Involved in This Project?

## What Precalculus Principles are Involved in This Project?

## Glossary of Functions
`encode_one_hot`:  

`add_bias`:  

`init_weight`:

`forward`:  

`predict`:  

`compute_loss`:  

`backward`:  

`gen_data`:  
