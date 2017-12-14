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

nb_epochs = 5
meta_lr = 0.00001
batch_size = 32

transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


class ModelSaver:

    def __init__(self):
        self.linear_weight = []

    def save_model(self, model):
        for name, w in model.named_parameters():
            if (name=="linear.weight"):
                self.linear_weight = w.data.clone()

    def undo_from_saved_model(self, model):
        model_dict = model.state_dict()
        saved_dict = {"linear.weight": self.linear_weight}
        model_dict.update(saved_dict)
        model.load_state_dict(model_dict)

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

    modelSaver = ModelSaver()

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
    
    for epoch in range(nb_epochs): # no of epochs

        acc = 0
        nfbs = 0
        long_running_loss = 0
        running_loss = 0
        nbs = 0

        for i, data in enumerate(trainloader, 0):

            optimizer = optim.SGD(model.parameters(), lr)

            inputs, labels = data

            inputs = inputs.view(batch_size, input_dims)

            inputs, labels = Variable(inputs), Variable(labels) 
            optimizer.zero_grad()


            last_layer = model.forward(inputs)
            output = loss.forward(last_layer, labels)
            output.backward()
            optimizer.step()

            #saving model
            modelSaver.save_model(model)

            #getting grad_t

            for name, w in model.named_parameters():
                if(name=="linear.weight"):
                    grad_t = w.grad

            optimizer.zero_grad()

            last_layer_next = model.forward(inputs)
            output_next = loss.forward(last_layer_next, labels)
            output_next.backward()


            #getting grad_t_plus_1

            for name, w in model.named_parameters():
                if(name=="linear.weight"):
                    grad_t_plus_1 = w.grad

            # undo saved model
            modelSaver.undo_from_saved_model(model)


            # time.sleep(0.5)

            grad_t = grad_t.view(-1,1)
            grad_t_plus_1 = grad_t_plus_1.view(-1,1)

            lr_grad = - torch.dot(grad_t, grad_t_plus_1)

            # print('lr grad is:', lr_grad)

            lr = lr - meta_lr*lr_grad.data[0]


            # print('grad_t and grad_t_plus_1 size')
            # print(grad_t.size())
            # print(grad_t_plus_1.size())
            
            running_loss += output.data[0]
            nbs +=1
            if(i%100) == 0:   
                print('[%d, %5d] loss: %.3f, lr: %f' % (epoch + 1, i + 1, running_loss / nbs, lr))
                running_loss = 0.0
                nbs = 0

            #getting the predictions on current training batch
            if (model.forward(inputs).data.numpy()).shape[0] == batch_size:
                nfbs += 1
                long_running_loss += output.data[0]
                preds = np.argmax((model.forward(inputs).data.numpy()).reshape((batch_size,n_classes)), axis=1)
                partial_acc = sum(preds == labels.data.numpy())
                acc += partial_acc

        #1-loss on whole training set
        training_loss.append(long_running_loss/nfbs)
        print('Training loss on this epoch: {}'.format(long_running_loss/(nfbs)))

        #2-accuracy on whole training set
        acc /= (nfbs*batch_size)
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
        print('Test loss on this epoch: {}'.format(np.mean(np.array(t_losses))))

        #4-accuracy on whole test set
        acc /= (nbs*batch_size)
        print('Test accuracy on this epoch: {}'.format(acc))
        test_acc_values.append(acc)

train(1)