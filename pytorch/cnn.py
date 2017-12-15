import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim


#PARAMETERS
n_classes = 10

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.saved_model2 = {}
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def save_model(self):
        self.saved_model = {
            "conv1.weight": self.conv1.weight.data.clone(),
            "conv1.bias": self.conv1.bias.data.clone(),
            "conv2.weight": self.conv2.weight.data.clone(),
            "conv2.bias": self.conv2.bias.data.clone(),
            "fc1.weight": self.fc1.weight.data.clone(),
            "fc1.bias": self.fc1.bias.data.clone(),
            "fc2.weight": self.fc2.weight.data.clone(),
            "fc2.bias": self.fc2.bias.data.clone(),
            "fc3.weight": self.fc3.weight.data.clone(),
            "fc3.bias": self.fc3.bias.data.clone()
        }

    def undo_using_saved_model(self):
        self.conv1.weight.data = self.saved_model['conv1.weight']
        self.conv1.bias.data = self.saved_model['conv1.bias']
        
        self.conv2.weight.data = self.saved_model['conv2.weight']
        self.conv2.bias.data = self.saved_model['conv2.bias']

        self.fc1.weight.data= self.saved_model['fc1.weight']
        self.fc1.bias.data= self.saved_model['fc1.bias']
        
        self.fc2.weight.data= self.saved_model['fc2.weight']
        self.fc2.bias.data= self.saved_model['fc2.bias']
        
        self.fc3.weight.data= self.saved_model['fc3.weight']
        self.fc3.bias.data= self.saved_model['fc3.bias']

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        d1 = nn.Dropout(p=0.5)
        x = d1(x)
        x = F.relu(self.fc2(x))
        d2 = nn.Dropout(p=0.5)
        x = d2(x)
        x = self.fc3(x)
        return x
