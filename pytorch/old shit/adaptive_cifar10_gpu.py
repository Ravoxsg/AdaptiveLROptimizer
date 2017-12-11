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


dtype = torch.cuda.FloatTensor

#HYPER-PARAMETERS
nb_epochs = 20
bs = 32 #batch size
eps = 1e-7 #epsilon value in the finite differences
sm_value = 0 #additive smoothing parameter
lr_ini = 0.1 #initial learning rate value
alpha = 0 #momentum parameter
criterion = nn.CrossEntropyLoss() #loss function
model_name = 'cnn_adaptive_20.pt' #model name

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

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = Variable(inputs).type(dtype), Variable(labels).type(torch.cuda.LongTensor)
            # zeroing gradient buffers
            net.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            #loss backward prop (only once, want to use same gradients to update learning rate)
            loss.backward()
            # saving model
            net.save_model2()

            # #loss
            # optimizer = optim.SGD(net.parameters(), lr=lr)
            # optimizer.step()
            # outputs = net(inputs)
            # loss = criterion(outputs, labels)

            # net.undo_using_saved_model2()

            #loss1
            optimizer = optim.SGD(net.parameters(), lr=lr+eps)
            optimizer.step()
            outputs_1 = net(inputs)
            loss1 = criterion(outputs_1, labels)

            net.undo_using_saved_model2()

            #loss2
            optimizer = optim.SGD(net.parameters(), lr=lr-eps)
            optimizer.step()
            outputs_2 = net(inputs)
            loss2 = criterion(outputs_2, labels)

            net.undo_using_saved_model2()

            #loss3
            optimizer = optim.SGD(net.parameters(), lr=lr+2*eps)
            optimizer.step()
            outputs_3 = net(inputs)
            loss3 = criterion(outputs_3, labels)
            
            net.undo_using_saved_model2()

            #loss4
            optimizer = optim.SGD(net.parameters(), lr=lr-2*eps)
            optimizer.step()
            outputs_4 = net(inputs)
            loss4 = criterion(outputs_4, labels)

            net.undo_using_saved_model2()

            #make actual update

            #learning rate
            finite_diff = (loss1 - loss2)/(loss3 + loss4 - 2*loss+sm_value)
            #print(loss1-loss2)
            #print(finite_diff)
            delta = alpha*delta - 2*eps*finite_diff.data[0]
            lr = lr + delta
            lr_values.append(lr)

            #weights
            # make_update(net, lr)
            optimizer = optim.SGD(net.parameters(), lr=lr)
            optimizer.step()

            #keeping track of everything
            running_loss += loss.data[0]
            if i % 156 == 155:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 156))
                print('learning rate:', lr)
                num = loss1 - loss2 
                print('Numerator: {}'.format(num.data[0]))
                denom = loss3 + loss4 - 2*loss
                print('Denominator: {}'.format(denom.data[0]))
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

    net = Net()
    net.cuda()

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