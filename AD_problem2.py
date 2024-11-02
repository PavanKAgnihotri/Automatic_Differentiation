#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:53:22 2024

@author: pavan
"""

#L(x) ≜ xT Jx + hT x,
import torch

n= 4
J = torch.randn(n,n)
h = torch.randn(n)
x = torch.randn(n)

x.requires_grad_(True)
objective = torch.matmul(x.T,torch.matmul(J, x)) + torch.matmul(h.T, x)
objective.backward()
grad_values = x.grad
manual_grad = torch.matmul(x.T, J+ J.T) + h.T

if torch.norm(grad_values - manual_grad).item() < 1e-4:
    print('partial derivative formula is correct.')
else:
    print('partial derivative formula is incorrect')
