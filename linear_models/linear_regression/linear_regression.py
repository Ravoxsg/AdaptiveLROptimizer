# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:37:00 2017

@author: mathi

Basic linear regression in 2 dimensions, with a randomly generated dataset.
We compare how GD and adaptive-LR GD perform on this dataset. 
"""

import numpy as np
import random
import matplotlib.pyplot as plt

#parameters
x_len = 100
num_of_steps = 5000 #number of iterations for both algorithms 
learning_rate = 0.00001 #learning rate in the normal GD
epsilon = 0.000001 #finite differences step
beta = 0.9

#linear regression parameters to learn
a = random.randint(-100,100)
print('a: {}'.format(a))
b = random.randint(-10,10)
print('b: {}'.format(b))

#linear function to learn
def linear_f(x):
    return a*x+b

#generate data
X_true = np.array(range(x_len))
bias = np.ones(x_len)
X = np.transpose(np.array([bias,X_true]))
Y_true = []
for i in range(x_len):
    Y_true.append(linear_f(i)+random.randint(-500,500))
Y_true = np.array(Y_true)
Y = np.expand_dims(Y_true, axis=1)

#cost_function
def L2(ytrue,w):
    ypred = np.matmul(X,w)
    return np.sum((ytrue-ypred)*(ytrue-ypred))/2*len(ytrue)

#gradient via finite differences
def grad(f,w):
    return (f(w+epsilon)-f(w-epsilon))/(2*epsilon)

#normal full gradient descent
w = np.array([0,0])
w = np.expand_dims(w,axis=1)
#print(w)
normal_costs = []
for i in range(num_of_steps):
    cost = L2(Y,w)
    normal_costs.append(cost)
    y_hat = np.matmul(X,w)
    #if (i%50 == 0): 
        #print("Cost:", cost)
    gradient = -((Y - y_hat).T.dot(X).reshape(-1,1))/Y.shape[0]
    #print(gradient)
    #print(w)
    w = w - learning_rate*gradient    
print('learned coefficients in normal GD: {},{}'.format(w[1],w[0]))

normal_w = w

#learning the learning rate with finite differences
w = np.array([0,0])
w = np.expand_dims(w,axis=1)
lr = 0.0000001
saved_lr = [lr]
new_costs = []
for i in range(num_of_steps):
    cost = L2(Y, w)
    new_costs.append(cost)
    y_hat = np.matmul(X,w)
    #if (i%50 == 0): print("Cost:", cost)
    gradient = -((Y - y_hat).T.dot(X).reshape(-1,1))/Y.shape[0]
    #print(gradient)
    #print(w)
    old_w = w
    w = w - lr*gradient
    lr = lr - 2*epsilon*beta*(L2(Y,old_w-(lr+epsilon)*gradient) - L2(Y,old_w-(lr-epsilon)*gradient) + epsilon)/(epsilon+L2(Y,old_w-(lr+2*epsilon)*gradient)+L2(Y,old_w-(lr-2*epsilon)*gradient)-2*L2(Y,old_w-lr*gradient))    
    if (i%50 == 0): 
        #print("Learned LR:", lr)
        saved_lr.append(lr)
print('learned coefficients in adaptive-LR GD: {},{}'.format(w[1],w[0]))

plt.plot(normal_costs[0:500])
plt.title('Cost over iterations in normal gradient descent')
plt.show()
plt.plot(new_costs[0:500])
plt.title('Cost over itrations in adaptive-LR gradient descent')
plt.show()

#predictions
plt.scatter(X_true,Y_true)
yprednormal = np.matmul(X,normal_w)
plt.scatter(X_true,yprednormal)
ypred = np.matmul(X,w)
plt.scatter(X_true,ypred)
plt.title('Regressions')
plt.show()

plt.plot(saved_lr)
plt.title('Learning rate value over iterations')
plt.show()
