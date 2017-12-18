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
bs = 256
eps = 1e-5
sm_value = 1e-6
lr_ini = 0.01
meta_lr = 0.000001
alpha = 0
criterion = nn.CrossEntropyLoss()
model = 'resnet'
model_name = 'cnn_adaptive_fo_{}_{}_{}.pt'.format(eps,nb_epochs,-int(np.log10(lr_ini))) #model name


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



def make_iterations(net, lr):

    training_loss = []
    test_loss = []
    train_acc_values = []
    test_acc_values = []
    lr_values = []
    delta = 0
    
    net.zero_grad()

    for epoch in range(nb_epochs): # no of epochs

        acc = 0
        nfbs = 0
        long_running_loss = 0
        running_loss = 0
        nbs = 0
        current_lrs = []

        for i, data in enumerate(trainloader, 0):

            # wrap them in Variable
            inputs, labels = data
            inputs, labels = Variable(inputs).type(dtype), Variable(labels).type(torch.cuda.LongTensor)

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

            lr = lr - meta_lr * (delta.data[0]/(2*eps))
            lr = max(sm_value,lr)
            current_lrs.append(lr)
            running_loss += loss.data[0]

            if i % 50 == 0:    # print every 2000 mini-batches
                print('loss 1:', loss1.data[0])
                print('loss 2:', loss2.data[0])
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 156))
                print('learning rate:', lr)
                print('Numerator: {}'.format(delta.data[0]))
                print('-----------------------------')
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
        train_loss = running_loss/nfbs
        training_loss.append(train_loss)
        print('Training loss on this epoch: {}'.format(train_loss))

        #2-accuracy on whole training set
        train_acc = acc/(nfbs*bs)
        print('Training accuracy on this epoch: {}'.format(train_acc))
        train_acc_values.append(train_acc)

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
                preds = np.argmax(((net(inputs).cpu()).data.numpy()).reshape((bs,10)), axis=1)
                partial_acc = sum(preds == (labels.cpu()).data.numpy())
                acc += partial_acc

        test_acc_values.append(acc) 

        #3-loss on whole test set
        test_l = np.mean(np.array(t_losses))
        test_loss.append(test_l)
        print('Test loss on this epoch: {}'.format(test_l))

        #4-accuracy on whole test set
        test_acc = acc/(nbs*bs)
        print('Test accuracy on this epoch: {}'.format(test_acc))
        test_acc_values.append(test_acc) 

        #5-learning rate
        current_lr = np.mean(np.array(current_lrs))
        lr_values.append(current_lr)
        print('Learning rate on this epoch: {}'.format(current_lr))

        #save to csv
        with open('results/partial_results_fo.csv','a') as file:
            file.write(str(train_acc)+','+str(train_loss)+','+str(test_acc)+','+str(test_l)+','+str(current_lr)+'\n')
            file.close()

    #save model
    torch.save(net.state_dict(),'models/'+model_name)


    return lr_values, running_loss_values


if __name__ == '__main__':

    if model == 'resnet':
        net = resnet.resnet18()
    else:
        net = cnn.Net()

    net.cuda()

    print(net)

    training_loss, train_acc_values, test_loss, test_acc_values, lr_values = make_iterations(net, lr_ini)

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

    plt.plot(lr_values)
    plt.title('Learning rate over iterations')
    plt.show()
