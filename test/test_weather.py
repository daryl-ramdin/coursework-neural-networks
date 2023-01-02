from src.weathercnn import WeatherCNN
from src.weathercnn import WeatherDataset
from src.weathercnn import preview_images
from src.weathercnn import predict
from src.weathercnn import get_accuracy
import matplotlib.pyplot as plt
import torch as torch


import torchvision.transforms as transforms
from  torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import os as os
from datetime import datetime
import numpy as np
import pandas as pd
import csv as csv
from sklearn.model_selection import train_test_split
from PIL import Image as PilImage
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import one_hot
from torch.nn.functional import log_softmax
from torch.nn.functional import cross_entropy


data_dir = "../data/weather"

'''
Let's create the WeatherDataset. The following transforms are applied
1) Resize the images to 128*128 pixels
2) Convert them to a Tensor
'''
transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
#Create the dataset
weather_ds = WeatherDataset("weather_annotations.csv","../data/weather",transform)

#Let's preview some of the data. We'll use a data loader for it
preview_loader = DataLoader(weather_ds,25, shuffle=True)
preview_images(preview_loader)

def test_2_layer_network():
    #Let's train our CNN

    #Create our CNN
    model = WeatherCNN(2)

    print("Model Definition:")
    print(model)

    #We next split our dataset into training and test datasets
    train_dataset, test_dataset = random_split(weather_ds, [0.8, 0.2])

    # Create our data loader for getting the training images
    train_loader = DataLoader(train_dataset, 128, shuffle = True)
    test_loader =  DataLoader(test_dataset, 128, shuffle = True)

    #Set the number of epochs
    epochs = 10

    #Set our learning rate
    learning_rate = 0.01

    #We use Adam as our optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    master_metrics_log = np.empty((0, 4), float)


    for epoch in range(epochs):
        print("Epoch:", epoch)
        start = datetime.now()
        model.train()
        train_losses = []
        for batch in train_loader:
            images, labels = batch
            #Forward pass
            out = model(images)
            #Calculate the loss and accuracy
            loss = cross_entropy(out, labels)   # Calculate loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #Let's see how our model is performing on the test data set
        loss,accuracy = predict(model, test_loader)
        # After each epoch print Loss and Accuracy
        print("Epoch", epoch, "Loss:", loss, "Accuracy: ", accuracy, "Duration:", datetime.now() - start)
        master_metrics_log = np.append(master_metrics_log, np.array([["Adam", epoch, loss, accuracy]]), 0)


    #Let's train our CNN

    #Create our CNN
    model = WeatherCNN()

    #We next split our dataset into training and test datasets
    train_dataset, test_dataset = random_split(weather_ds, [0.8, 0.2])

    # Create our data loader for getting the training images
    train_loader = DataLoader(train_dataset, 128, shuffle = True)
    test_loader =  DataLoader(test_dataset, 128, shuffle = True)

    #Set the number of epochs
    epochs = 10

    #Set our learning rate
    learning_rate = 0.01

    #We use Adam as our optimizer
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    #master_metrics_log = np.empty((0, 4), float)

    for epoch in range(epochs):
        print("Epoch:", epoch)
        start = datetime.now()
        model.train()
        train_losses = []
        for batch in train_loader:
            images, labels = batch
            #Forward pass
            out = model(images)
            #Calculate the loss and accuracy
            loss = cross_entropy(out, labels)   # Calculate loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #Let's see how our model is performing on the test data set
        loss,accuracy = predict(model, test_loader)
        # After each epoch print Loss and Accuracy
        print("Epoch", epoch, "Loss:", loss, "Accuracy: ", accuracy, "Duration:", datetime.now() - start)
        master_metrics_log = np.append(master_metrics_log, np.array([["SGD", epoch, loss, accuracy]]), 0)

    plt.figure(figsize=(16,12))
    plt.plot(master_metrics_log[master_metrics_log[:,0]=="SGD"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="SGD"][:, 2].astype(float),label="SGD")
    plt.plot(master_metrics_log[master_metrics_log[:,0]=="Adam"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="Adam"][:, 2].astype(float),label="Adam")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(16,12))
    plt.plot(master_metrics_log[master_metrics_log[:,0]=="SGD"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="SGD"][:, 3].astype(float),label="SGD")
    plt.plot(master_metrics_log[master_metrics_log[:,0]=="Adam"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="Adam"][:, 3].astype(float),label="Adam")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
def test_3_layer_network():
    #Let's train the CNN

    #Create the CNN
    model = WeatherCNN(3)

    print("Model Definition:")
    print(model)

    #We next split our dataset into training and test datasets
    train_dataset, test_dataset = random_split(weather_ds, [0.8, 0.2])

    # Create our data loader for getting the training images
    train_loader = DataLoader(train_dataset, 128, shuffle = True)
    test_loader =  DataLoader(test_dataset, 128, shuffle = True)

    #Set the number of epochs
    epochs = 10

    #Set our learning rate
    learning_rate = 0.01

    #We use Adam as our optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    master_metrics_log = np.empty((0, 4), float)


    for epoch in range(epochs):
        print("Epoch:", epoch)
        start = datetime.now()
        model.train()
        train_losses = []
        for batch in train_loader:
            images, labels = batch
            #Forward pass
            out = model(images)
            #Calculate the loss and accuracy
            loss = cross_entropy(out, labels)   # Calculate loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #Let's see how our model is performing on the test data set
        loss,accuracy = predict(model, test_loader)
        # After each epoch print Loss and Accuracy
        print("Epoch", epoch, "Loss:", loss, "Accuracy: ", accuracy, "Duration:", datetime.now() - start)
        master_metrics_log = np.append(master_metrics_log, np.array([["Adam", epoch, loss, accuracy]]), 0)


    #Let's train our CNN

    #Create our CNN
    model = WeatherCNN()

    #We next split our dataset into training and test datasets
    train_dataset, test_dataset = random_split(weather_ds, [0.8, 0.2])

    # Create our data loader for getting the training images
    train_loader = DataLoader(train_dataset, 128, shuffle = True)
    test_loader =  DataLoader(test_dataset, 128, shuffle = True)

    #Set the number of epochs
    epochs = 10

    #Set our learning rate
    learning_rate = 0.01

    #We use Adam as our optimizer
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    #master_metrics_log = np.empty((0, 4), float)

    for epoch in range(epochs):
        print("Epoch:", epoch)
        start = datetime.now()
        model.train()
        train_losses = []
        for batch in train_loader:
            images, labels = batch
            #Forward pass
            out = model(images)
            #Calculate the loss and accuracy
            loss = cross_entropy(out, labels)   # Calculate loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #Let's see how our model is performing on the test data set
        loss,accuracy = predict(model, test_loader)
        # After each epoch print Loss and Accuracy
        print("Epoch", epoch, "Loss:", loss, "Accuracy: ", accuracy, "Duration:", datetime.now() - start)
        master_metrics_log = np.append(master_metrics_log, np.array([["SGD", epoch, loss, accuracy]]), 0)

    plt.figure(figsize=(16,12))
    plt.plot(master_metrics_log[master_metrics_log[:,0]=="SGD"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="SGD"][:, 2].astype(float),label="SGD")
    plt.plot(master_metrics_log[master_metrics_log[:,0]=="Adam"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="Adam"][:, 2].astype(float),label="Adam")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(16,12))
    plt.plot(master_metrics_log[master_metrics_log[:,0]=="SGD"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="SGD"][:, 3].astype(float),label="SGD")
    plt.plot(master_metrics_log[master_metrics_log[:,0]=="Adam"][:, 1].astype(float), master_metrics_log[master_metrics_log[:,0]=="Adam"][:, 3].astype(float),label="Adam")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

test_2_layer_network()
test_3_layer_network()