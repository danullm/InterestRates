#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 09:38:12 2018

@author: daniel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
###############################################################################

T = [0, 2, 3, 4, 5, 7, 10, 20, 30]
Y = [0, -.79, -.73, -.65, -.55, -.33, -.04, .54, .73]

#Y = [0, 0.472	, 0.701,	0.925, 1.138,	 1.516, 1.941	, 2.44, 2.37]

Y = [x/100 for x in Y]


###############################################################################
###############################################################################

def baseH(data, u, i):
    i = i + 1
    Ti = data.iloc[i,].ttm  
    tmp = Ti*(1+u)-u**2/2
    ref = Ti*(1+Ti)-Ti**2/2
    tmp[u>Ti] = ref
    return(tmp)

def forwardCurve(data, u, beta):
    beta0 = beta[0]
    betas = beta[1:]    
    N = len(betas)
    basisfunctions = [baseH(data, u,i)*betas[i] for i in range(N)]
    f = beta0 + sum(basisfunctions)
    return(f)

def skalarprod(data, i,j):
    Ti = data.iloc[i,].ttm
    Tj = data.iloc[j,].ttm
    Tk = min(Ti, Tj)

    return(Ti*Tj + Ti*Tj*Tk - (Ti+Tj)/2*Tk**2 + 1/3*Tk**3)

def getYield(u,f,T):
    forward = pd.DataFrame(data = f, index = u)
    dt = np.diff(u)[0]
    tmp = forward[forward.index <= T].sum()/T*dt
    return(float(tmp))

def getLibor(t,f,T):
    y = getYield(t,f,T)
    return((np.exp(T*y)-1)/T)

def Lormier(T,Y, alpha, retval = 'yield', dt = 1./12):
    data = pd.DataFrame({'ttm': T, 'Y': Y})

    M = np.zeros([len(T)]*2)
    y = np.zeros(len(T))

    M_tmp = np.zeros([len(T)-1]*2)
    
    for i in range(len(M_tmp)):
        for j in range(len(M_tmp)):
            M_tmp[i,j] = skalarprod(data, i+1,j+1)*alpha
    
    M_tmp += np.diag([1]*(len(T)-1))
    M[1:,1:] = M_tmp
    
    for i in range(len(T)):
        if i > 0:
            y[i] = alpha * data.iloc[i].Y * data.iloc[i].ttm
            M[i,0] = data.iloc[i].ttm*alpha
            M[0,i] = data.iloc[i].ttm
    
    
    beta = np.dot(np.linalg.inv(M), y)
    
    t = np.arange(dt,max(T)+dt,dt)
    
    f = forwardCurve(data, t,beta)
    
    yields = [getYield(t,f,x) for x in t]

    if retval == 'yield':
        return(pd.DataFrame(data = yields, index = t))
    else:
        return(pd.DataFrame(data = f, index = t))
###############################################################################
###############################################################################
    
    
yields1 = Lormier(T,Y, 1)
yields2 = Lormier(T,Y, 0.10)
yields3 = Lormier(T,Y, 0.01)

plt.scatter(T[1:],Y[1:])
plt.plot(yields1, label = 'alpha = 1')
plt.plot(yields2, label = 'alpha = 0.1')
plt.plot(yields3, label = 'alpha = 0.01')
plt.grid()
plt.legend()
plt.tight_layout()

dt = 1./12

t = np.arange(dt, max(T)+dt, dt)

f = Lormier(T = T, Y = Y, alpha = 0.1, retval = 'forward', dt = dt)

libor = getYield(t,f,6)

print('The answer to the question is {:.2%}'.format(libor))
