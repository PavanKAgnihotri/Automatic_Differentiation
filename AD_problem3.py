#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:12:42 2024

@author: pavan
"""

import torch
J = torch.FloatTensor([[1,-2,1],[2,3,-4],[1,-4,1]]) # J must be in size 3 x 3
h =  torch.FloatTensor([[1],[1],[1]]) # h must be in size 3 x 1 
x0 = torch.FloatTensor([[0.1],[0.1],[0.1]]) # x0 must be in size 3 x 1
alpha = 1e-3
num_epochs = 10000
print(J.size()) 
print(h.size())
print(x0.size())
x = x0

for epoch in range(num_epochs): 
    print('Epoch', epoch)
# Create a variable for x
    x_var = torch.nn.Parameter(x, requires_grad = True)
# Compute the objective L
    obj = torch.matmul(torch.transpose(x_var, 0, 1),torch.matmul(J, x_var)) + torch.matmul(torch.transpose(h,0,1), x_var)
# Compute the gradient by automatic differentiation
    obj.backward()
# Gradient Descent update
    grad = x_var.grad
    x = x - alpha * grad
# Make sure that x is in range [-1, 1]
    x[x < -1] = -1
    x[x > 1] = 1

print('--------------')
print('Solution:', x)
