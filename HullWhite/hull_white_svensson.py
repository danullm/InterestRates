#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:45:53 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sys
sys.path.insert(0, '/home/daniel/Seafile/Dani/Python/InterestRates/Svensson/')
sys.path.insert(0, '/home/daniel/Seafile/Dani/Python/InterestRates/Vasicek/')
from svensson import *
from vasicek import *

#------------------------------------------------------------------------------

def hull_white_forward(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, T = 10, N = 50):
    t = np.linspace(0,T,N)
    phi = phi_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, t)
    vasicek = vasicek_A_prime(kappa, theta, sigma, t) + vasicek_B_prime(kappa, t)*r0
    f0 = phi + vasicek
    f0 = pd.DataFrame(data = f0, index = t)
    return(f0)

def phi_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, t):
    f0 = svensson_forwards(beta0, beta1, beta2, beta3, tau1, tau2, t)
    vasicek = vasicek_A_prime(kappa, theta, sigma, t) + vasicek_B_prime(kappa, t)*r0
    phi2 = f0 - vasicek
    return(phi2)
    
def phi_prime_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, t):
    T1 = t/tau1
    T2 = t/tau2
    svensson = -beta1/tau1*np.exp(-T1) + beta2/tau1 * np.exp(-T1) - beta2/tau1*T1*np.exp(-T1)
    svensson += beta3/tau2 * np.exp(-T2) - beta3/tau2*T2*np.exp(-T2)

    svensson += - vasicek_A_2prime(kappa, theta, sigma, t) - vasicek_B_2prime(kappa, t)*r0
    
    return(svensson)

def theta_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, t):
    tmp = phi_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, t)
    tmp += phi_prime_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, t)/kappa
    return(tmp)
    

def hw_process_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, T = 1., N = 100, seed = 0):
    if seed != 0:
        np.random.seed(seed)

    dt = T/N
    t = 0.
    rates = [r0]
    for i in range(N):
        dr = kappa*(theta_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, t)-rates[-1])*dt
        dr += sigma*np.random.normal(size = 1, scale = np.sqrt(dt))
        dr = float(dr)
        t += dt
        rates.append(rates[-1] + dr)
    
    return(pd.DataFrame(data = rates, index = [x*dt for x in range(N+1)] ))


if __name__ == '__main__':
    
    plot = True
       
    [beta0, beta1, beta2, beta3, tau1, tau2] = \
        np.array([ 1.60913/100, -2.25762/100, -2.88699/100,  0.02519/100,  2.17563,  0.18767])
    
    [r0, theta, kappa, sigma] = np.array([-0.01020897,  0.17611249,  0.12063917,  0.18981454])
    
    
    if plot == True: 
        T = 10
        t = np.arange(0,T,1)
        
        f0 = svensson_forwards(beta0, beta1, beta2, beta3, tau1, tau2, t)
        vasicek_fit = vasicek_forward_curve(r0, theta, kappa, sigma, T)
        hw_fit = hull_white_forward(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, T)
        
        
        plt.scatter(t,f0, label = 'Market', marker = '+', c = 'red') 
        plt.plot(vasicek_fit, label = 'Vasicek')
        plt.plot(hw_fit, label = 'Hull White')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        
    
    T , N, mc = [25., 500, 20]
    
    rates = hw_process_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, T, N)
    for i in range(mc-1):
        rates = pd.concat([rates, hw_process_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, T, N)], axis = 1)


    rate = rates.iloc[:,0]
    rate.plot()
    plt.plot(np.exp(-rate.cumsum()*rate.index[1]))
    
