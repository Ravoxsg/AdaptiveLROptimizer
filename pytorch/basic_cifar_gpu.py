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
training_set_size = 50000
test_set_size = 10000
n_classes = resnet.n_classes
nb_epochs = 60
bs = 32 #batch size
bpetrain = int(training_set_size/bs) #number of batches to get full training set
lr = 0.01 #learning rate
criterion = nn.CrossEntropyLoss() #loss
model = 'cnn' #model to use
model_name = '{}_basic_{}_{}.pt'.format(model,nb_epochs,-int(np.log10(lr))) #model name

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if (n_classes == 10): 

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

if (n_classes == 100): 
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.xlabel('Random images')
    plt.show()

def make_iterations(net, lr):

    training_loss = []
    test_loss = []
    train_acc_values = []
    test_acc_values = []
    lr_values = []
    delta = 0

    optimizer = optim.SGD(net.parameters(), lr=lr)
    net.zero_grad()

    for epoch in range(nb_epochs): # no of epochs

        acc = 0
        nfbs = 0
        long_running_loss = 0
        running_loss = 0
        nbs = 0

        for i, data in enumerate(trainloader, 0):

            nbs += 1

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

            #keeping track of everything during training, every tenth of epoch
            running_loss += loss.data[0]
            if i % int(bpetrain/10) == 0:   
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / nbs))
                running_loss = 0.0
                nbs = 0

            #getting the predictions on current training batch
            if ((net(inputs).cpu()).data.numpy()).shape[0] == bs:
                nfbs += 1
                long_running_loss += loss.data[0]
                preds = np.argmax(((net(inputs).cpu()).data.numpy()).reshape((bs,n_classes)), axis=1)
                partial_acc = sum(preds == (labels.cpu()).data.numpy())
                acc += partial_acc

        #1-loss on whole training set
        train_loss = long_running_loss/nfbs
        training_loss.append(train_loss)
        print('Training loss on this epoch: {}'.format(train_loss))

        #2-accuracy on whole training set
        train_acc = acc/(nfbs*bs)
        print('Training accuracy on this epoch: {}'.format(train_acc))
        train_acc_values.append(train_acc)

        #Test set

        acc = 0
        nbs = 0
        t_losses = []

        for i, data in enumerate(testloader, 0):

            inputs, labels = data
            inputs, labels = Variable(inputs, volatile=True).type(dtype), Variable(labels, volatile=True)

            #getting the predictions on current batch
            outputs = net(inputs).cpu()
            if ((outputs.data.numpy()).shape[0] == bs):
                nbs += 1
                t_loss = criterion(outputs, labels)
                t_losses.append(t_loss.data[0])
                preds = np.argmax((outputs.data.numpy()).reshape((bs,n_classes)), axis=1)
                partial_acc = sum(preds == (labels.data.numpy()))
                acc += partial_acc

        #3-loss on whole test set
        test_l = np.mean(np.array(t_losses))
        test_loss.append(test_l)
        print('Test loss on this epoch: {}'.format(test_l))

        #4-accuracy on whole test set
        test_acc = acc/(nbs*bs)
        print('Test accuracy on this epoch: {}'.format(test_acc))
        test_acc_values.append(test_acc) 

        #save to csv
        with open('results/basic_partial_results.csv','a') as file:
            file.write(str(train_acc)+','+str(train_loss)+','+str(test_acc)+','+str(test_l)+'\n')
            file.close()

    #save model
    torch.save(net.state_dict(),'models/'+model_name)

    return training_loss, train_acc_values, test_loss, test_acc_values


if __name__ == '__main__':
    
    if model == 'resnet':
        net = resnet.resnet18()
    else:
        net = cnn.Net()

    net.cuda()

    print(net)

    training_loss, train_acc_values, test_loss, test_acc_values = make_iterations(net, lr)

    plt.plot(training_loss)
    plt.ylim((0,5))
    plt.title('Training loss over iterations')
    plt.show()

    plt.plot(test_loss)
    plt.ylim((0,5))
    plt.title('Test loss over iterations')
    plt.show()

    plt.plot(train_acc_values)
    plt.ylim((0,1))
    plt.title('Training accuracy over iterations')
    plt.show()

    plt.plot(test_acc_values)
    plt.ylim((0,1))
    plt.title('Test accuracy over iterations')
    plt.show()

