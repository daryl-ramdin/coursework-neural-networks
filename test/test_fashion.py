import torch as torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
import os as os
import numpy as np
import pandas as pd
import csv as csv
from sklearn.model_selection import train_test_split
from PIL import Image as PilImage
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from torch.utils.data import  random_split
from torch.nn.functional import log_softmax
from datetime import datetime
from torch.nn.functional import cross_entropy
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
from src.fashioncnn import FashionCNN
from src.fashioncnn import FashionDataset
from src.fashioncnn import preview_images
from src.fashioncnn import get_accuracy


df = pd.read_csv("../data/fashion/fashion-mnist_train.csv")

# Get the labels
train_labels = df["label"].to_numpy()

# Get the image data
train_images = df[df.columns[df.columns != 'label']].to_numpy()

# Conver the images and labels to tensors
train_images_tensor = torch.tensor(train_images) / 255 # This is to normalise the data
train_labels_tensor = torch.tensor(train_labels)

# Let's create our FashionDataset
fashion_ds = FashionDataset(train_images_tensor,train_labels_tensor)

#Let's preview the data. We create a loader to assist
preview_loader = DataLoader(fashion_ds,25, shuffle=True)
preview_images(preview_loader)

# Let's split our data into training and test data
train_dataset,test_dataset =  random_split(fashion_ds,[0.8,0.2])

# Create our data loader for getting the training images
train_loader = DataLoader(train_dataset, 10, shuffle = True)
test_loader =  DataLoader(test_dataset, 10, shuffle = True)


def test_optimizers_epochs():
    master_metrics_log = np.empty((0, 4), float)



    #Create Model  instance
    model = FashionCNN()
    #For this we will use CrossEntropyLoss with a SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    num_epochs = 10
    # Train the model
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        print("Epoch:",epoch)
        start = datetime.now()
        accuracy_log = []
        loss_log = []
        accuracy = 0
        loss = 0
        for batch in train_loader:
            #Get the images and labels
            images, labels = batch
            # Forward pass
            outputs = model(images)
            #Calculate the accuracy and loss
            accuracy = get_accuracy(outputs,labels)
            loss = cross_entropy(outputs, labels)
            accuracy_log.append(accuracy)
            loss_log.append(loss)
            # Back propagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #After each epoch print Loss and Accuracy
        accuracies = torch.stack(accuracy_log).mean().item() * 100
        losses = torch.stack(loss_log).mean().item()
        master_metrics_log = np.append(master_metrics_log, np.array([["SGD", epoch, losses, accuracies]]), 0)
        print("Completed epoch:", epoch, "Accuracy:", accuracies, "Loss:", losses, "in:", datetime.now() - start)


    #Create Model  instance
    model = FashionCNN()
    #For this we will use CrossEntropyLoss with a SGD optimizer
    lossanalyzer = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    # Train the model


    for epoch in range(num_epochs):
        print("Epoch:",epoch)
        start = datetime.now()
        accuracy_log = []
        loss_log = []
        accuracy = 0
        loss = 0
        for batch in train_loader:
            #Get the images and labels
            images, labels = batch
            # Forward pass
            outputs = model(images)
            #Calculate the accuracy and loss
            accuracy = get_accuracy(outputs,labels)
            loss = lossanalyzer(outputs, labels)
            accuracy_log.append(accuracy)
            loss_log.append(loss)
            # Back propagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #After each epoch print Loss and Accuracy
        print("Completed epoch:", epoch, "in:",datetime.now() - start)
        accuracies = torch.stack(accuracy_log).mean().item() * 100
        losses = torch.stack(loss_log).mean().item()
        master_metrics_log = np.append(master_metrics_log,np.array([["Adam",epoch,losses,accuracies]]),0)

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


test_optimizers_epochs()


