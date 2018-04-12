#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:39:15 2018

@author: daniel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


C = pd.read_excel('bootstrap.xls','pseudoinverse', skiprows = 27, usecols = [3 + x for x in range(39)])
p = pd.read_excel('bootstrap.xls','pseudoinverse', skiprows = 27, usecols = [0])

today = [pd.Timestamp('10/03/2012')]
dates = list(C)
today.extend(dates)

C = C.as_matrix()
p = np.array([float(x) for x in p.P])

deltas = np.diff(today)
deltas = [x.days/360. for x in deltas]

vec = np.zeros(len(deltas))
vec[0] = 1.
vec[1:] = 0.

W = np.diag(1./np.sqrt(deltas))
M = np.diag([1]*len(deltas))

for i in range(len(M)-1):
    M[i+1,i] = -1
    
Mm1 = np.linalg.inv(M)
Wm1 = np.linalg.inv(W)

A = np.dot(C , np.dot(Mm1 , Wm1))
A_m = np.dot(A.transpose(), np.linalg.inv(np.dot(A , A.transpose())))

delta = np.dot(A_m, (p - np.dot(C , np.ones(len(deltas)) )))

plt.plot(delta)

d_tmp = np.dot(Wm1, delta) + vec
d = np.dot(Mm1, d_tmp)

plt.plot(d)

answer = (dates[-1]-dates[-2]).days/360.*(d[-2]/d[-1]-1)
print('The answer to the question is {:.2%}'.format(answer))



