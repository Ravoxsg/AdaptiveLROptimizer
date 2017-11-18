import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

nb_epochs = 2
eps = 1e-6
sm_value = 1e-9
# lr = 0.00001

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.xlabel('Random images')
    plt.show()


class Net(nn.Module):

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


    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def make_iterations(net, lr):
    running_loss_values = []
    lr_values = []
    running_loss = 0
    net.zero_grad()
    for epoch in range(nb_epochs): # no of epochs
        for i, data in enumerate(trainloader, 0):
            if (i%100==0): 
                print("i is:", i)
                print('lr: ', lr)
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # zeroing gradient buffers
            net.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.data[0]
            if (i%100 ==0):
                print('loss: ', loss.data[0])
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                print('learning rate:', lr)
                running_loss_values.append(running_loss)

            #loss backward prop (only once, want to use same gradients to update learning rate)
            loss.backward()
            # saving model
            net.save_model()

            # make_update(net, lr + eps)
            optimizer = optim.SGD(net.parameters(), lr=lr + eps)
            optimizer.step()
            outputs_1 = net(inputs)
            loss1 = criterion(outputs_1, labels)

            net.undo_using_saved_model()

            # make_update(net, lr - eps)
            optimizer = optim.SGD(net.parameters(), lr=lr - eps)
            optimizer.step()
            outputs_2 = net(inputs)
            loss2 = criterion(outputs_2, labels)

            net.undo_using_saved_model()

            # make_update(net, lr + 2*eps)
            optimizer = optim.SGD(net.parameters(), lr=lr + 2*eps)
            optimizer.step()
            outputs_3 = net(inputs)
            loss3 = criterion(outputs_3, labels)
            
            net.undo_using_saved_model()

            # make_update(net, lr - 2*eps)
            optimizer = optim.SGD(net.parameters(), lr=lr - 2*eps)
            optimizer.step()
            outputs_4 = net(inputs)
            loss4 = criterion(outputs_4, labels)

            net.undo_using_saved_model()

            delta = (loss1 - loss2 + sm_value)/(loss3 + loss4 - 2*loss + sm_value)

            #make actual update

            # make_update(net, lr)
            optimizer = optim.SGD(net.parameters(), lr=lr)
            optimizer.step()
            lr = lr - 2*eps*delta.data[0]
            lr_values.append(lr)
    return lr_values, running_loss_values

def make_update(net, lr):

    net.conv1.weight.data -= net.conv1.weight.grad.data.float()*lr
    net.conv1.bias.data -= net.conv1.bias.grad.data.float()*lr
    
    net.conv2.weight.data -= net.conv2.weight.grad.data.float()*lr
    net.conv2.bias.data -= net.conv2.bias.grad.data.float()*lr

    net.fc1.weight.data -= net.fc1.weight.grad.data.float()*lr
    net.fc1.bias.data -= net.fc1.bias.grad.data.float()*lr

    net.fc2.weight.data -= net.fc2.weight.grad.data.float()*lr
    net.fc2.bias.data -= net.fc2.bias.grad.data.float()*lr

    net.fc3.weight.data -= net.fc3.weight.grad.data.float()*lr
    net.fc3.bias.data -= net.fc3.bias.grad.data.float()*lr



net = Net()

print(net)

lr_values, running_loss_values = make_iterations(net, 0.000001)

plt.plot(lr_values)
plt.show()
