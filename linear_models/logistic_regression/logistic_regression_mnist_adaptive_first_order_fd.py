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

nb_epochs = 20
meta_lr = 0.00001
batch_size = 32
eps = 1e-5
bpetrain = int(60000/batch_size)
sm_value = 1e-6
alpha = 0 # momentum coefficient on the LR

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
    criterion = torch.nn.CrossEntropyLoss(size_average=True)
    
    for epoch in range(nb_epochs): # no of epochs

        print("EPOCH:", epoch)
        acc = 0
        nfbs = 0
        long_running_loss = 0
        running_loss = 0
        nbs = 0

        current_lrs = []

        for i, data in enumerate(trainloader, 0):

            nbs += 1

            inputs, labels = data

            inputs = inputs.view(batch_size, input_dims)
            inputs, labels = Variable(inputs), Variable(labels) 


            model.zero_grad()

            modelSaver.save_model(model)

            # loss 1
            optimizer = optim.SGD(model.parameters(), lr=(lr + eps))
            last_layer = model.forward(inputs)
            loss = criterion.forward(last_layer, labels)
            loss.backward()
            optimizer.step()
            outputs = model.forward(inputs)
            loss1 = criterion.forward(outputs, labels)

            modelSaver.undo_from_saved_model(model)
            model.zero_grad()

            # loss 2
            optimizer = optim.SGD(model.parameters(), lr = (lr-eps))
            last_layer = model.forward(inputs)
            loss = criterion.forward(last_layer, labels)
            loss.backward()
            optimizer.step()
            outputs = model.forward(inputs)
            loss2 = criterion.forward(outputs, labels)

            modelSaver.undo_from_saved_model(model)
            model.zero_grad()

            # new loss
            optimizer = optim.SGD(model.parameters(), lr = lr)
            last_layer = model.forward(inputs)
            loss = criterion.forward(last_layer, labels)
            loss.backward()
            optimizer.step()
            outputs = model.forward(inputs)
            loss_wt_1 = criterion.forward(outputs, labels)


            delta = loss1 - loss2

            lr = lr - meta_lr * (delta.data[0]/(2*eps))
            lr = max(sm_value,lr)
            current_lrs.append(lr)
            running_loss += loss.data[0]

            if i % int(bpetrain/10) == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / nbs))
                print('learning rate:', lr)
                print('-----------------------------')
                running_loss = 0.0
                nbs = 0


            #getting the predictions on current training batch
            if (model.forward(inputs).data.numpy()).shape[0] == batch_size:
                nfbs += 1
                long_running_loss += loss.data[0]
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
                t_loss = criterion.forward(outputs, labels)
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

        #5-learning rate
        lr_values.append(np.mean(np.array(current_lrs)))
        learning_rate_on_epoch = np.mean(np.array(current_lrs))
        print('Learning rate on this epoch: {}'.format(np.mean(np.array(current_lrs))))

        with open('results_first_order.csv', 'a') as f:
            running_str = ''
            running_str += str(training_loss_on_epoch) + ","
            running_str += str(training_accuracy_on_epoch) + ","
            running_str += str(test_loss_on_epoch) + ","
            running_str += str(test_accuracy_on_epoch) + ","
            running_str += str(learning_rate_on_epoch) + "\n"
            f.write(running_str)

    return training_loss, train_acc_values, test_loss, test_acc_values, lr_values

if __name__ == '__main__':

    training_loss, train_acc_values, test_loss, test_acc_values, lr_values = train(0.1)

    print(training_loss, train_acc_values, test_loss, test_acc_values, lr_values)
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