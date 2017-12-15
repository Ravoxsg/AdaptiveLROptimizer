import torch
import numpy as np
import csv
import gc
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
THC_CACHING_ALLOCATOR=0

#HYPER-PARAMETERS
training_set_size = 50000
test_set_size = 10000
n_classes = resnet.n_classes
nb_epochs = 50
bs = 256 # batch size
bpetrain = int(training_set_size/bs) #number of batches to get full training set
eps = 1e-5 # finite differences step
sm_value = 1e-6 # denominator smoothing in the finite differences formula
lr_ini = 0.01 # initial learning rate
alpha = 0 # momentum coefficient on the LR
criterion = nn.CrossEntropyLoss() # loss
model = 'resnet' # model to use
model_name = '{}_adaptive_{}_{}_{}.pt'.format(model,eps,nb_epochs,-int(np.log10(lr_ini))) #model name
use_val = False #whether or not to use validation set to get loss values

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

val_len = int(len(trainloader)*0.2)
if use_val == False:
    val_len = 0


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
    
    net.zero_grad()

    #building the validation set
    val_data = []
    val_labels = []

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs, labels = Variable(inputs).type(dtype), Variable(labels).type(torch.cuda.LongTensor)
        val_data.append(inputs)
        val_labels.append(labels)        

        if (i >= val_len):
            break

    for epoch in range(nb_epochs): # no of epochs

        acc = 0
        nfbs = 0
        long_running_loss = 0
        running_loss = 0
        nbs = 0
        current_lrs = []

        for i, data in enumerate(trainloader, val_len):

            nbs += 1

            inputs, labels = data
            inputs, labels = Variable(inputs).type(dtype), Variable(labels).type(torch.cuda.LongTensor)

            if (use_val == True):
                idx = np.random.randint(low = 0, high = val_len-1, size = 1)
                val_batch_data = val_data[idx[0]]
                val_batch_labels = val_labels[idx[0]]
                w_inputs = val_batch_data
                w_labels = val_batch_labels
            else:
                w_inputs = inputs
                w_labels = labels

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
            outputs = net(w_inputs)
            loss1 = criterion(outputs, w_labels)

            net.undo_using_saved_model()
            net.zero_grad()

            #loss2
            optimizer = optim.SGD(net.parameters(), lr = (lr-eps))
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            outputs = net(w_inputs)
            loss2 = criterion(outputs, w_labels)

            net.undo_using_saved_model()
            net.zero_grad()                        

            #loss3
            optimzer = optim.SGD(net.parameters(), lr=(lr+2*eps))
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            outputs = net(w_inputs)
            loss3 = criterion(outputs, w_labels)

            net.undo_using_saved_model()
            net.zero_grad()

            #loss4
            optimizer = optim.SGD(net.parameters(), lr=(lr-2*eps))
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            outputs = net(w_inputs)
            loss4 = criterion(outputs, w_labels)

            net.undo_using_saved_model()
            net.zero_grad()

            #new loss
            optimizer = optim.SGD(net.parameters(), lr = lr)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            outputs = net(w_inputs)
            loss_wt_1 = criterion(outputs, w_labels)

            #learning rate
            numerator = loss1 - loss2
            denominator = loss3 + loss4 -2*loss_wt_1
            if (denominator.data[0] == 0.0):
                denominator += sm_value
            finite_diff = numerator/denominator

            delta = alpha*delta - 2*eps*finite_diff.data[0]
            lr = max(sm_value,lr + delta)
            current_lrs.append(lr)

            #keeping track of everything during training, every tenth of epoch
            running_loss += loss.data[0]
            if i % int(bpetrain/10) == 0:   
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / nbs))
                print('learning rate:', lr)
                print('Numerator: {}'.format(numerator.data[0]))
                print('Denominator: {}'.format(denominator.data[0]))
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

            gc.collect()
            del inputs, labels

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

            gc.collect()
            del inputs, labels

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
        with open('results/partial_results.csv','a') as file:
            file.write(str(train_acc)+','+str(train_loss)+','+str(test_acc)+','+str(test_l)+','+str(lr)+'\n')
            file.close()

    #save model
    torch.save(net.state_dict(),'models/'+model_name)

    return training_loss, train_acc_values, test_loss, test_acc_values, lr_values


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



