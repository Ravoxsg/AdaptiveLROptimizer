# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:37:00 2017

@author: mathi

Basic linear regression in d dimensions where d is a parameter to choose, with a randomly generated dataset.
Dimension here represents the degree of the polynom in x.
We compare how GD and adaptive-LR GD perform on this dataset. 
"""

import numpy as np
import random
import matplotlib.pyplot as plt

#parameters
x_len = 30 #number of points
num_of_steps = 5000 #iterations
learning_rate = 0.00000000000000000001 #learning rate of normal GD - very small to ensure convergence !!
epsilon = 0.000001 #step in the finite differences
beta = 0.9
deg = 20 #dimension

#linear regression parameters to learn
coeff = []
for j in range(deg+1):
    coeff.append(random.randint(-10,10))
print('coefficients: {}'.format(coeff))

#linear function to learn
def linear_f(x):
    r = 0
    for j in range(len(coeff)):
        r += coeff[j]*np.power(x,j)
    return r

#generate data
X_true = np.array(range(x_len))
X = np.zeros(shape=(x_len,deg+1))
for j in range(deg+1):
    X[:,j] = np.power(X_true,j)
Y_true = []
for i in range(x_len):
    Y_true.append(linear_f(i)+random.randint(-50,50))
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
w = np.zeros(deg+1)
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
print('learned coefficients in normal GD: {}'.format(w))

normal_w = w

#learning the learning rate with finite differences
w = np.zeros(deg+1)
w = np.expand_dims(w,axis=1)
lr = 0.000000001
saved_lr = [lr]
new_costs = []
print(w)
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
    lr = lr - 2*epsilon*(L2(Y,old_w-(lr+epsilon)*gradient) - L2(Y,old_w-(lr-epsilon)*gradient) + epsilon)/(epsilon+L2(Y,old_w-(lr+2*epsilon)*gradient)+L2(Y,old_w-(lr-2*epsilon)*gradient)-2*L2(Y,old_w-lr*gradient))    
    if (i%50 == 0): 
        #print("Learned LR:", lr)
        saved_lr.append(lr)
print('learned coefficients in adaptive-LR GD: {}'.format(w))

plt.plot(normal_costs[0:500])
plt.title('Cost over iteration in normal GD')
plt.show()
plt.plot(new_costs[0:500])
plt.title('Cost over iteration in adaptive-LR GD')
plt.show()

#predictions
plt.plot(X_true,Y_true)
plt.title('Polynomial to fit')
plt.show()

yprednormal = np.matmul(X,normal_w)
plt.plot(X_true,yprednormal)
plt.title('Regression with normal GD')
plt.show()

ypred = np.matmul(X,w)
plt.plot(X_true,ypred)
plt.title('Regression with adaptive-LR GD')
plt.show()

plt.plot(saved_lr)
plt.title('Learning rate over itrations')
plt.show()
