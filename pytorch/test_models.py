#Get the accuracy of the trained neural network classifier

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim

from cnn import Net


bs = 16
model_name = 'cnn_adaptive_7.pt'

#load model
model = Net()
model.load_state_dict(torch.load('models/'+model_name))

#data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)


def test_model(model):

	acc = 0

	for i, data in enumerate(testloader, 0):

		
		inputs, labels = data
		inputs, labels = Variable(inputs), Variable(labels)
		#print(labels)
		preds = np.argmax((model(inputs).data.numpy()).reshape((bs,10)), axis=1)
		partial_acc = sum(preds == labels.data.numpy())
		acc += partial_acc

	acc /= (i*bs)

	return acc


if __name__ == '__main__':

	accuracy = test_model(model)
	print(accuracy)