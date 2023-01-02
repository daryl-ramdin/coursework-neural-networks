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



'''
This is the dataset for loading the fashion data. We inherit from the 
pytorch dataset
'''
class FashionDataset(Dataset):
    def __init__(self,images_tensor,labels_tensor,transforms=None):
        #We reshape our images to a nx1x28x28 array. 1 is for 1 channel
        self.images = images_tensor.reshape(images_tensor.shape[0], 1, 28, 28)
        self.labels = labels_tensor
        self.transform = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #Get the image and label at the given index
        image = self.images[index]
        label = self.labels[index]
        if self.transform != None:
            image = self.transform(image)

        #Return the image and label
        return image, label

'''
This is the Convolution Neural Network class.
'''
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_unit_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Second unit of convolution
        self.conv_unit_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten())

        # Fully connected layers
        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.conv_unit_1(x)
        out = self.conv_unit_2(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = log_softmax(out, dim=1)
        return out

'''
This function allows us to see a subset of the images.
Code Reference: INM702 2022, LAB09_4
https://moodle.city.ac.uk/pluginfile.php/2994042/mod_folder/content/0/Lab09_4.ipynb?forcedownload=1
'''
def preview_images(preview_dl):
    """Plot images grid of single batch"""
    for batch in preview_dl:
        images, labels = batch
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=5).permute(1, 2, 0)) #Convert channelxWxH to WxHxchannel
        plt.show()
        break

'''
This function helps to calculate the accuracy
Code Reference: INM702 2022, LAB09_4
'''
def get_accuracy(outputs, labels):
    #As we are using softmax, get the column with the largest value. This
    #will correspond to the prediction
    values, indices = torch.max(outputs, dim=1)

    #Get the total number of rows that match and divide by the number of
    #indices to give an average accuracy
    return torch.tensor(torch.sum(indices == labels).item() / len(indices))
