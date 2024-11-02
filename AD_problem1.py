#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:45:26 2024

@author: pavan
"""

import torch

x = torch.tensor(1.0, requires_grad = True)
y = torch.tensor(2.0, requires_grad = True)
z = torch.tensor(3.0, requires_grad = True)

#f(x,y,z)=x2y+y2z+z2 +4
def fun(x,y,z):
    return x**2 * y + y**2 * z + z**2 + 4

forward_value = fun(x, y, z)

print('Forward Pass: ',forward_value.item())
forward_value.backward()
print('partial derivative of f w.r.t x:',x.grad.item())
print('partial derivative of f w.r.t y:',y.grad.item())
print('partial derivative of f w.r.t z:',z.grad.item())
backward_value = x.grad+y.grad+z.grad
print('Backward Pass:',backward_value.item())
