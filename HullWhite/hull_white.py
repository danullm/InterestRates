#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:45:53 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    return(t)

if __name__ == '__main__':
    
    [beta0, beta1, beta2, beta3, tau1, tau2] = \
        np.array([ 1.60913/100, -2.25762/100, -2.88699/100,  0.02519/100,  2.17563,  0.18767])
    
    [r0, theta, kappa, sigma] = np.array([-0.01020897,  0.17611249,  0.12063917,  0.18981454])
    
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
