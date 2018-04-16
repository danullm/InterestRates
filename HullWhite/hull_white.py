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
    phi = phi_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta2, tau1, tau2, t)
    vasicek = vasicek_A_prime(kappa, theta, sigma, t) + vasicek_B_prime(kappa, t)*r0
    f0 = phi + vasicek
    f0 = pd.DataFrame(data = f0, index = t)
    return(f0)

def phi_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, t):
    f0 = svensson_forwards(beta0, beta1, beta2, beta3, tau1, tau2, t)
    vasicek = vasicek_A_prime(kappa, theta, sigma, t) + vasicek_B_prime(kappa, t)*r0
    phi2 = f02 - vasicek
    return(phi2)
    



if __name__ == '__main__':
    
    [beta0, beta1, beta2, beta3, tau1, tau2] = \
        np.array([ 1.60913/100, -2.25762/100, -2.88699/100,  0.02519/100,  2.17563,  0.18767])
    
    [r0, theta, kappa, sigma] = np.array([-0.01235011,  0.01870044,  0.2178629 ,  0.01442659])
    
    t = np.arange(1./12,10,1./12)

    f0 = svensson_forwards(beta0, beta1, beta2, beta3, tau1, tau2, t)
    
    vasicek_fit = vasicek_forward_curve(r0, theta, kappa, sigma, T = max(t))
    vasicek = vasicek_A_prime(kappa, theta, sigma, t) + vasicek_B_prime(kappa, t)*r0
    
    phi = phi_svensson(r0, theta, kappa, sigma, beta0, beta1, beta2, beta3, tau1, tau2, t)
    
    
    plt.scatter(t,f0, label = 'Market') 
    
    plt.plot(t,vasicek, label = 'Vasicek')
    plt.plot(vasicek_fit, label = 'Vasicek2')
    #plt.plot(hw_fit, label = 'Hull White')
    plt.legend()
    plt.grid()
    plt.tight_layout()
