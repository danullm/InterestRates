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
    tmp = np.array([phi_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta2, tau1, tau2, T) for T in t])
    tmp += np.array([vasicek_A_prime(kappa, sigma, T) + vasicek_B_prime(kappa, T)*r0 for T in t])
    tmp = pd.DataFrame(data = tmp, index = t)
    
    return(tmp)


def phi_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, t):
    f0 = svensson_forwards(beta0, beta1, beta2, beta3, tau1, tau2, t)/100
    vasicek = vasicek_A_prime(kappa, sigma, t) + vasicek_B_prime(kappa, t)*r0
    return(f0-vasicek)
    
#def phi_svensson_partial(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, t):
    

if __name__ == '__main__':
    
    [beta0, beta1, beta2, beta3, tau1, tau2] = np.array([ 1.60913, -2.25762, -2.88699,  0.02519,  2.17563,  0.18767])
    [r0, theta, kappa, sigma] = np.array([-0.01020908,  0.02677791,  0.12066193,  0.00350423])
    
    t = np.arange(1./12,10,1./12)

    fc = [svensson_forwards(beta0, beta1, beta2, beta3, tau1, tau2, x)/100 for x in t]

    vasicek_fit = vasicek_forward_curve(r0, theta, kappa, sigma, T = max(t))
    hw_fit = hull_white_forward(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, T = max(t))
    
    plt.scatter(t,fc, label = 'Market', marker = '+', alpha = 0.5) 
    plt.plot(vasicek_fit, label = 'Vasicek') 
    plt.plot(hw_fit, label = 'Hull White')
    plt.legend()

