#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:22:45 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from mpl_toolkits.mplot3d import Axes3D

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def hw_process_spline(r0, kappa, sigma, forward_spl, T = 5., N = 100, seed = 0):
    
    if seed != 0:
        np.random.seed(seed)
    dt = T/N
    dW = np.random.normal(size = N, scale = np.sqrt(dt))
    t = 0.
    rates = [r0]
    for i in range(N):
        dr = kappa*( theta_hw(forward_spl, kappa, sigma, t) / kappa - rates[-1] )*dt
        dr += sigma*dW[i]
        dr = float(dr)
        t += dt
        rates.append(rates[-1] + dr)
    return(pd.DataFrame(data = rates, index = [x*dt for x in range(N+1)] ))


def theta_hw(forward_spl, kappa, sigma, t):
    tmp = forward_spl.derivative()(t)
    tmp += forward_spl(t)*kappa
    tmp += sigma**2/2/kappa*(1-np.exp(-2*kappa*t))
    return(tmp)


def hw_B(kappa, T, t=0.):
    if t > 0:
        T = T-t
    tmp = 1 - np.exp(-kappa*T)
    tmp = tmp / kappa
    return(tmp)
    

def hw_A(kappa, sigma, forward_spl, T, t = 0.):
    P0T = np.exp(-forward_spl.integral(0,T))
    P0t = np.exp(-forward_spl.integral(0,t))
    B = hw_B(kappa, T, t)
    tmp = B*forward_spl(t) 
    tmp = tmp - sigma**2/4/kappa*B**2*(1-np.exp(-2*kappa*t))
    A = P0T/P0t
    A = A*np.exp(tmp)
    return(A)
    
    
def hw_P(kappa, sigma, forward_spl, r0, T, t = 0.):
    return( float(hw_A(kappa, sigma, forward_spl, T, t) * np.exp( -hw_B(kappa, T, t) * r0 ) ) ) 
    
    
def forward_integration(forward_spl, x):
    return(forward_spl.integral(0, x))
    
v_forward_integration = np.vectorize(forward_integration)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == '__main__':
    
    t = np.arange(0,11)
    f = [.005, .0075, .0135, .01755, .0202, .02136, .02285, .02372, .02482, .02561, .02608]

    forward_spl = InterpolatedUnivariateSpline(t, f)
    r0 = forward_spl(0)
    
    T = max(t)
    N = T*100
    mc = 10
    
    kappa = 0.022
    sigma = 0.009
    
    xnew = np.linspace(0, T, N)
    
    market_discounts = np.exp(-v_forward_integration(forward_spl, t))
    model_discounts = [hw_P(kappa, sigma, forward_spl, r0, T, 0) for T in xnew]
   
    rates = hw_process_spline(r0, kappa, sigma, forward_spl, T, N)
    for i in range(mc-1):
        rates = pd.concat([rates, hw_process_spline(r0, kappa, sigma, forward_spl, T, N)], axis = 1)
        
        
    rate = rates.iloc[:,0]
    
    time = np.linspace(0, T, N+1)   
    tenor = np.linspace(0, max(t), max(t)+1)   
    
    plot_df = pd.DataFrame()
    
    for ts in time:
        
        r0 = rate.loc[ts,]
        
        discount_curve = [hw_P(kappa, sigma, forward_spl, r0, T, 0) for T in tenor]
        yield_curve = -np.log(discount_curve)/tenor
        yield_curve[0] = r0
        
        df = pd.DataFrame({'time': [ts]*len(yield_curve), 'tenor': tenor, 'yield': yield_curve})        
    
        plot_df = pd.concat([plot_df, df])
    
    plot_df.reset_index(inplace = True, drop = True)

    
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(xnew, model_discounts, label = 'model')
    ax.scatter(t, market_discounts, marker = '+', c = 'red', label = 'market')
    ax.set_xlabel('Tenor')
    ax.set_ylabel('instantaneous forward rate')
    ax.legend()

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax = fig.gca(projection='3d')
    ax.plot_trisurf(plot_df['time'], plot_df['tenor'], plot_df['yield'], cmap=plt.cm.viridis, linewidth=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Tenor')
    ax.set_zlabel('Yield')
    
    plt.tight_layout()
    plt.show()