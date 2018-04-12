#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:26:12 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def CIR_process(r0, theta, kappa, sigma, T = 1., N = 100, seed = 0):
    if seed != 0:
        np.random.seed(seed)

    dt = T/N
    rates = [r0]
    for i in range(N):
        dr = kappa*(theta-rates[-1])*dt + sigma*np.sqrt(rates[-1])*np.random.normal(size = 1, scale = np.sqrt(dt))
        rates.append(rates[-1] + dr)
    
    return(pd.DataFrame(data = rates, index = [x*dt for x in range(N+1)] ))




if __name__ == '__main__':

    r0, theta, kappa, sigma = [0.06, 0.08, 0.86, 0.01]
    
    T , N, mc = [10., 200, 5]
    
    plot = False
    
    rates = CIR_process(r0, theta, kappa, sigma, T, N)
    for i in range(mc-1):
        rates = pd.concat([rates, CIR_process(r0, theta, kappa, sigma, T, N)], axis = 1)

    
    if plot == True:
    
        plt.plot(rates, alpha = 0.25)
        plt.axhline(theta, c = 'black', linestyle = ':')
        plt.grid()
        plt.tight_layout()
