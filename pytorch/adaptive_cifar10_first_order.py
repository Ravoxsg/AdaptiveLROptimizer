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


nb_epochs = 20
bs = 32
eps = 1e-3
sm_value = 1e-10
lr_ini = 1
meta_lr = 0.0001
alpha = 0
criterion = nn.CrossEntropyLoss()
model_name = 'cnn_adaptive_fo_20_0_001.pt' #model name


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



def make_iterations(net, lr):
    running_loss_values = []
    lr_values = []
    train_acc_values = []
    test_acc_values = []
    running_loss = 0
    net.zero_grad()

    for epoch in range(nb_epochs): # no of epochs

        acc = 0
        nbs = 0
        losses_vars = []

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            net.zero_grad()

            net.save_model()


            optimizer = optim.SGD(net.parameters(), lr = (lr+eps))
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()             # we not have g_t
            optimizer.step()            # we now have w_t = w_t - (lr+eps)*g_t
            outputs = net(inputs)
            loss1 = criterion(outputs, labels)

            net.undo_using_saved_model()    #back to w_t
            net.zero_grad()

            optimizer = optim.SGD(net.parameters(), lr = (lr-eps))
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()             # we now have g_t
            optimizer.step()            # we not have w_t = w_t - (lr-eps)*g_t
            outputs = net(inputs)
            loss2 = criterion(outputs, labels)

            net.undo_using_saved_model()
            net.zero_grad()

            optimizer = optim.SGD(net.parameters(), lr = lr)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()             # we now have g_t
            optimizer.step()            # we now have w_t = w_t - lr*g_t

            delta = loss1 - loss2

            lr = lr - meta_lr * (delta.data[0]/2*eps)
            lr_values.append(lr)
            running_loss += loss.data[0]

            if i % 50 == 0:    # print every 2000 mini-batches
                print('loss 1:', loss1.data[0])
                print('loss 2:', loss2.data[0])
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 156))
                print('learning rate:', lr)
                print('Numerator: {}'.format(delta.data[0]))
                print('-----------------------------')
                running_loss_values.append(running_loss/156)
                running_loss = 0.0

            #getting the predictions on current batch
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
            inputs, labels = Variable(inputs), Variable(labels)

            #getting the predictions on current batch
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


    return lr_values, running_loss_values


if __name__ == '__main__':

    net = Net()
    # net.cuda()

    print(net)

    lr_values, running_loss_values, train_acc_values, test_acc_values = make_iterations(net, lr_ini)

    plt.plot(running_loss_values)
    plt.ylim((0,5))
    plt.title('Loss over iterations')
    plt.show()

    plt.plot(lr_values)
    plt.ylim((0,20*lr_ini))
    plt.title('Learning rate over iterations')
    plt.show()

    plt.plot(train_acc_values)
    plt.ylim((0,1))
    plt.title('Training accuracy over iterations')
    plt.show()

    plt.plot(test_acc_values)
    plt.ylim((0,1))
    plt.title('Test accuracy over iterations')
    plt.show()