This repository contains the code for various coursework projects on Neural Networks while doing my masters degree at City, University of London. They are as follows:

Built from scratch Neural Network using numpy: neuralnetwork.py
A convolutional neural network built with Pytorch for training on the MINST Fashion Dataset: fashioncnn.py
A convolutional neural network built with Pytorch for predicting the weather from a dataset of images: weathercnn.py
An evaluation of Dijkstra's algorithm over a random and simple search for finding the shortest path between two points. No neural network used.


Directions:

The first step is to install the necessary libraries.
The following would normally have to be installed

pip install numpy
pip install scikit-learn
pip install torch
pip install torchvision

The next step is to create a data folder in the root directory. It should be at the same level
as the src folder so that the directory looks like

data
notebooks
src
test

The next step is to create the following subfolders in the data folder
fashion
weather

The next step is to download an unzip the weather dataset. It can be found at https://www.kaggle.com/datasets/jehanbhathena/weather-dataset
and should be unzipped into a weather subfolder in the data folder.
The structure of the weather folder should then be:
dew
fogsmog
frost
glaze.
.
.
.
snow

The Fashion-MNIST dataset can be found here: https://www.kaggle.com/datasets/zalando-research/fashionmnist?select=fashion-mnist_test.csv
IT should be downloaded and unzipped into the fashion sub folder in the data folder

The annotation file has already been included but a new one can be created
by running the create_annotations function in the file weathercnn.py

To test the various solutions, the test directory contains the test scripts that were used
to generate the report results. The files are as follows:

test_fashion - script for the Fashion-MNIST
test_mycnn - script for the custom Neural Network
test_shortestpath - script for testing the search algorithms
test_weather - script for testing the weather dataset

There are also Jupyter notebooks that run part of the scripts to allow for the visualization
of some of the data. These are located in the notebooks directory and are as follows:
fashion.ipynp           -   Notebook for the Fashion-MNIST dataset
my_neural_network.ipynp -   Notebook for the custom Neural Network
search_algorithms       -   Notebook for the search algorithms
weather.ipynb           -   Notebook for the weather dataset
