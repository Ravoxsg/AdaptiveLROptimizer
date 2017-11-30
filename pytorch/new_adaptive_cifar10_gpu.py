import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim

import resnet
import cnn

dtype = torch.cuda.FloatTensor


#HYPER-PARAMETERS
nb_epochs = 20
bs = 32 #batch size
eps = 1e-5 #finite differences step
sm_value = 1e-6 #denominator smoothing in the finite differences formula
lr_ini = 0.001 #initial learning rate
alpha = 0 #momentum coefficient on the LR
criterion = nn.CrossEntropyLoss() #loss
model = 'resnet' #model to use
model_name = '{}_adaptive_{}_{}_{}.pt'.format(model,eps,nb_epochs,-int(np.log10(lr_ini))) #model name

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

def make_iterations(net, lr):
    running_loss_values = []
    lr_values = []
    train_acc_values = []
    test_acc_values = []
    running_loss = 0
    delta = 0
    net.zero_grad()

    for epoch in range(nb_epochs): # no of epochs

        acc = 0
        nbs = 0
        losses_vars = []

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = Variable(inputs).type(dtype), Variable(labels).type(torch.cuda.LongTensor)

            # zeroing gradient buffers
            net.zero_grad()

            # saving model
            net.save_model()
            
            #loss1
            optimizer = optim.SGD(net.parameters(), lr = (lr+eps))
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            outputs1 = net(inputs)
            loss1 = criterion(outputs1, labels)

            net.undo_using_saved_model()
            net.zero_grad()

            #loss2
            optimizer = optim.SGD(net.parameters(), lr = (lr-eps))
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            outputs = net(inputs)
            loss2 = criterion(outputs, labels)

            net.undo_using_saved_model()
            net.zero_grad()                        

            #loss3
            optimzer = optim.SGD(net.parameters(), lr=(lr+2*eps))
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimzer.step()
            outputs = net(inputs)
            loss3 = criterion(outputs, labels)

            net.undo_using_saved_model()
            net.zero_grad()

            #loss4
            optimizer = optim.SGD(net.parameters(), lr=(lr-2*eps))
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            outputs = net(inputs)
            loss4 = criterion(outputs, labels)

            net.undo_using_saved_model()
            net.zero_grad()

            #new loss
            optimizer = optim.SGD(net.parameters(), lr = lr)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            outputs = net(inputs)
            loss_wt_1 = criterion(outputs, labels)

            #checking out the losses
            # print("loss: {}".format(loss.data[0]))
            # print("loss1: {}".format(loss1.data[0]))
            # print("loss2: {}".format(loss2.data[0]))
            # print("loss3: {}".format(loss3.data[0]))
            # print("loss4: {}".format(loss4.data[0]))
            # print("loss_new: {}".format(loss_wt_1.data[0]))

            #learning rate
            numerator = loss1 - loss2
            denominator = loss3 + loss4 -2*loss_wt_1
            if (denominator.data[0] == 0.0):
                denominator += sm_value
            finite_diff = numerator/denominator

            delta = alpha*delta - 2*eps*finite_diff.data[0]
            #print(delta.data[0])
            lr = lr + delta
            lr_values.append(lr)

            #keeping track of everything
            running_loss += loss.data[0]
            if i % 156 == 155:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 156))
                print('learning rate:', lr)
                print('Numerator: {}'.format(numerator.data[0]))
                print('Denominator: {}'.format(denominator.data[0]))
                print('-----------------------------')
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

    return lr_values, running_loss_values, train_acc_values, test_acc_values


if __name__ == '__main__':

    if model == 'resnet':
        net = resnet.resnet18()
    else:
        net = cnn.Net()

    net.cuda()

    print(net)

    lr_values, running_loss_values, train_acc_values, test_acc_values = make_iterations(net, lr_ini)

    plt.plot(running_loss_values)
    plt.ylim((0,5))
    plt.title('Loss over iterations')
    plt.show()

    plt.plot(lr_values)
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