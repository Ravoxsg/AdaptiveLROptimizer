import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.datasets import fetch_mldata
import time


nb_epochs = 10

batch_size = 32

transform = transforms.ToTensor()

def load_data():

    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]

    train_data = torch.from_numpy(train_data)
    train_targets = torch.from_numpy(train_targets)
    test_data = torch.from_numpy(test_data)
    test_targets = torch.from_numpy(test_targets)


    return train_data, train_targets, test_data, test_targets


trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)




def build_model(input_dim, output_dim):

    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, output_dim, bias=False))
    return model

def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)

def train(lr):

    training_loss = []
    test_loss = []
    train_acc_values = []
    test_acc_values = []
    lr_values = []
    delta = 0

    n_classes = 10
    input_dims = trainset.train_data.size()[-1] * trainset.train_data.size()[-2]
    model = build_model(input_dims, n_classes)
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.SGD(model.parameters(), lr)

    for epoch in range(nb_epochs): # no of epochs

        acc = 0
        nfbs = 0
        long_running_loss = 0
        running_loss = 0
        nbs = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs = inputs.view(batch_size, input_dims)

            inputs, labels = Variable(inputs), Variable(labels) 
            optimizer.zero_grad()


            last_layer = model.forward(inputs)
            output = loss.forward(last_layer, labels)
            output.backward()
            optimizer.step()


            running_loss += output.data[0]
            nbs +=1
            if(i%100) == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / nbs))
                running_loss = 0.0
                nbs = 0

                for i,j in model.named_parameters():
                    print(j.norm())
                    print('---------------------')
                    # print()
                    print(j.grad.norm())
                # time.sleep(2)


            #getting the predictions on current training batch
            if (model.forward(inputs).data.numpy()).shape[0] == batch_size:
                nfbs += 1
                long_running_loss += output.data[0]
                preds = np.argmax((model.forward(inputs).data.numpy()).reshape((batch_size,n_classes)), axis=1)
                partial_acc = sum(preds == labels.data.numpy())
                acc += partial_acc

        #1-loss on whole training set
        training_loss.append(long_running_loss/nfbs)
        training_loss_on_epoch = long_running_loss/(nfbs)
        print('Training loss on this epoch: {}'.format(long_running_loss/(nfbs)))

        #2-accuracy on whole training set
        acc /= (nfbs*batch_size)
        training_accuracy_on_epoch = acc
        print('Training accuracy on this epoch: {}'.format(acc))
        train_acc_values.append(acc)
        

        acc = 0
        nbs = 0
        t_losses = []
        
        # testing
        for i, data in enumerate(testloader, 0):

            inputs, labels = data
            curr_batch_size = inputs.size()[0]
            inputs = inputs.view(curr_batch_size, input_dims)
            inputs, labels = Variable(inputs), Variable(labels)

            #getting the predictions on current batch
            outputs = model.forward(inputs)
            if ((outputs.data.numpy()).shape[0] == batch_size):
                nbs += 1
                t_loss = loss.forward(outputs, labels)
                t_losses.append(t_loss.data[0])
                preds = np.argmax((outputs.data.numpy()).reshape((batch_size,n_classes)), axis=1)
                partial_acc = sum(preds == (labels.data.numpy()))
                acc += partial_acc

        #3-loss on whole test set
        test_loss.append(np.mean(np.array(t_losses)))
        test_loss_on_epoch = np.mean(np.array(t_losses))
        print('Test loss on this epoch: {}'.format(np.mean(np.array(t_losses))))

        #4-accuracy on whole test set
        acc /= (nbs*batch_size)
        test_accuracy_on_epoch = acc
        print('Test accuracy on this epoch: {}'.format(acc))
        test_acc_values.append(acc)

        with open('results_basic.csv', 'a') as f:
            running_str = ''
            running_str += str(training_loss_on_epoch) + ","
            running_str += str(training_accuracy_on_epoch) + ","
            running_str += str(test_loss_on_epoch) + ","
            running_str += str(test_accuracy_on_epoch) + "\n"
            f.write(running_str)

    print(model.state_dict())
    print(model.state_dict().keys())
    print(dir(model.state_dict()))
    print('-------')
    for i,j in model.named_parameters():
        print(j.grad)

train(0.1)

