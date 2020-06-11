# Liam Kronman and Sofiane Larbi's Mathematical Project
Creating an **Artificial Neural Network (ANN)** through the lens of calculus and precalculus.

## How to Run Our Project
First, make sure you have python3 installed on your computer. It can be downloaded [here](https://www.python.org).  

Then type this line of code into your terminal, in the same directory that you have cloned this repository, to download necessary dataset packages:  

`pip install -r requirements.txt`  

Finally, run our primary python file in your terminal with this line:

`python3 mnist_nn.py`  


## What is an Artificial Neural Network (ANN)?
Imagine you have a bunch of images of dogs and cats, and you want to develop a program that will determine which images are of dogs and which are of cats. This may be harder than you think. Comparing the corresponding pixels of different images to a sort of "platonic ideal" of a dog or a cat is impractical, not only because the images might be of different dimensions, but also because if the pictured pet is not in the *exact* same position as your model cat/dog, you will probably get an incorrect output. For problems like these, that require reverse engineering the abstract categories of various inputs, building an **ANN** may be the most effective solution.  

An **ANN** is a series of layers made up of weighted nodes (called **neurons**) that strings together the pieces of an input. It returns which output category the input is most similar to after being passed through the whole "network." As an input is passed through the first layer to the layers that are not seen to the user (known as **hidden layers**), the various neurons are "activated" to alter the values of the input. When the input reaches the final layer, whichever node has the highest value, the category that is most similar to the input, is the one that is returned.

## How Does Our ANN Work?
Please check out the **Glossary of Functions** below for more info on the referenced functions.
Our

## What Calculus Principles are Involved in This Project?

## What Precalculus Principles are Involved in This Project?

## What precalculus principles are involved in this project?

## Glossary of Functions
`encode_one_hot`:  

`add_bias`:  

`init_weight`:

`forward`:  

`predict`:  

`compute_loss`:  

`backward`:  

`gen_data`:  
