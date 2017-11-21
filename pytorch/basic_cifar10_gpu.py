import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim


dtype = torch.cuda.FloatTensor

#HYPER-PARAMETERS
nb_epochs = 1
bs = 32
lr = 0.001
criterion = nn.CrossEntropyLoss()
model_name = 'basic_1.pt'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.xlabel('Random images')
    plt.show()

class Net(nn.Module):

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
    train_acc_values = []
    test_acc_values = []
    running_loss = 0
    optimizer = optim.SGD(net.parameters(), lr=lr)
    net.zero_grad()

    for epoch in range(nb_epochs): # no of epochs

        acc = 0
        nbs = 0

        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).type(dtype), Variable(labels).type(torch.cuda.LongTensor)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 156 == 155:    # print every 150 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 156))
                running_loss_values.append(running_loss/156)
                running_loss = 0.0

            #getting the predictions on current training batch
            if ((net(inputs).cpu()).data.numpy()).shape[0] == bs:
                nbs += 1
                preds = np.argmax(((net(inputs).cpu()).data.numpy()).reshape((bs,10)), axis=1)
                partial_acc = sum(preds == (labels.cpu()).data.numpy())
                acc += partial_acc

        #accuracy on whole training set
        acc /= (nbs*bs)
        print('Training accuracy on this epoch: {}'.format(acc))
        train_acc_values.append(acc)

        acc = 0
        nbs = 0

        for i, data in enumerate(testloader, 0):

            inputs, labels = data
            inputs, labels = Variable(inputs).type(dtype), Variable(labels).type(torch.cuda.LongTensor)

            #getting the predictions on current test batch
            if ((net(inputs).cpu()).data.numpy()).shape[0] == bs:
                nbs += 1
                preds = np.argmax(((net(inputs).cpu()).data.numpy()).reshape((bs,10)), axis=1)
                partial_acc = sum(preds == (labels.cpu()).data.numpy())
                acc += partial_acc

        #accuracy on whole test set
        acc /= (nbs*bs)
        print('Test accuracy on this epoch: {}'.format(acc))
        test_acc_values.append(acc)

    #save model
    torch.save(net.state_dict(),'models/'+model_name)

    return running_loss_values, train_acc_values, test_acc_values


if __name__ == '__main__':
    
    net = Net()
    net.cuda()

    print(net)

    running_loss_values, train_acc_values, test_acc_values = make_iterations(net, lr)

    plt.plot(running_loss_values)
    plt.ylim((0,5))
    plt.title('Loss over iterations')
    plt.show()

    plt.plot(train_acc_values)
    plt.ylim((0,1))
    plt.title('Training accuracy over iterations')
    plt.show()

    plt.plot(test_acc_values)
    plt.ylim((0,1))
    plt.title('Test accuracy over iterations')
    plt.show()
