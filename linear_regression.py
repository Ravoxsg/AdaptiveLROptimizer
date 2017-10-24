# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:37:00 2017

@author: mathi
"""

import numpy as np
import random
import matplotlib.pyplot as plt

#parameters
x_len = 100
num_of_steps = 2000
learning_rate = 0.00001
epsilon = 0.00001

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

plt.scatter(X_true,Y_true)
    
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
print(w)
for i in range(num_of_steps):
    cost = L2(Y,w)
    y_hat = np.matmul(X,w)
    if (i%50 == 0): print("Cost:", cost)
    gradient = -((Y - y_hat).T.dot(X).reshape(-1,1))/Y.shape[0]
    #print(gradient)
    #print(w)
    w = w - learning_rate*gradient    
print('learned coefficients: {},{}'.format(w[1],w[0]))

#predictions
ypred = np.matmul(X,w)
plt.scatter(X_true,ypred)

#learning the learning rate with finite differences
w = np.array([0,0])
w = np.expand_dims(w,axis=1)
lr = 0.00001
print(w)
for i in range(num_of_steps):
    cost = L2(Y, w)
    y_hat = np.matmul(X,w)
    if (i%50 == 0): print("Cost:", cost)
    gradient = -((Y - y_hat).T.dot(X).reshape(-1,1))/Y.shape[0]
    #print(gradient)
    #print(w)
    old_w = w
    w = w - lr*gradient
    lr = lr - 2*epsilon*(L2(Y,old_w-(lr+epsilon)*gradient) - L2(Y,old_w-(lr-epsilon)*gradient) + epsilon)/(epsilon+L2(Y,old_w-(lr+2*epsilon)*gradient)+L2(Y,old_w-(lr-2*epsilon)*gradient)-2*L2(Y,old_w-lr*gradient))    
    if (i%50 == 0): print("Learned LR:", lr)
print('learned coefficients: {},{}'.format(w[1],w[0]))

#predictions
ypred = np.matmul(X,w)
plt.scatter(X_true,ypred)
