#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:34:12 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

from vasicek import *


def vasicek_convexity_adjustment_sympy(r0, theta, kappa, sigma, t, T0, T1):
    delta = T1 - T0
    PtT0 = vasicek_discount_factor(r0, theta, kappa, sigma, T0, t)
    PtT1 = vasicek_discount_factor(r0, theta, kappa, sigma, T1, t)
    
    kappaval = kappa
    sigmaval = sigma
    tval = t
    T0val = T0
    T1val = T1
    
    r0, theta, kappa, sigma = sp.symbols('r0 theta kappa sigma')
    T0, T1, t, s, u, v = sp.symbols('T0 T1 t s u v')

    vasicek_sigma = sigma*sp.exp(-kappa*(T0-t))

    int1 = vasicek_sigma.subs([(t, s), (T0, u)])
    int1 = sp.integrate(int1, (u, s, T1))
    
    int2 = vasicek_sigma.subs([(t, s), (T0, v)])
    int2 = sp.integrate(int2, (v, T0, T1))
    
    integral = int1*int2
    integral = sp.integrate(integral, (s, t, T0))
    
    integral = integral.subs([ (t, tval), (T0, T0val), (T1,T1val), (sigma,sigmaval), (kappa,kappaval) ]).evalf()
    
    gamma = PtT0/PtT1/delta
    gamma = gamma * (np.exp(float(integral)) - 1)

    return(gamma)

def vasicek_convexity_adjustment(r0, theta, kappa, sigma, t, T0, T1):
    delta = T1 - T0
    PtT0 = vasicek_discount_factor(r0, theta, kappa, sigma, T0, t)
    PtT1 = vasicek_discount_factor(r0, theta, kappa, sigma, T1, t)
    
    int1 = ( np.exp(-kappa*T0)-np.exp(-kappa*T1) )/kappa**3*sigma**2/2
    int1 *= np.exp(-kappa*T1)
    int1 *= ( np.exp(kappa*t) - np.exp(kappa*T0) )
    int1 *= ( np.exp(kappa*t) + np.exp(kappa*T0) - 2*np.exp(kappa*T1) )
    
    integral = sigma**2/kappa**3*np.exp(-kappa*T1)
    integral *= (np.exp(-kappa*T1) - np.exp(-kappa*T0))
    integral *= ( 0.5*(np.exp(2*kappa*T0) - np.exp(2*kappa*t)) + np.exp(kappa*T0) - np.exp(kappa*t) )

    gamma = PtT0/PtT1/delta
    gamma = gamma * (np.exp(int1) - 1)
    
    return(gamma)

if __name__ == '__main__':
    
    r0, theta, kappa, sigma = [0.06, 0.08, 0.86, 0.01]
    sigma = 0.01 
    
    t = 0.
    T0 = 0
    T1 = 1
    
    vasicek_convexity_adjustment(r0, theta, kappa, sigma, t, T0, T1)*100*100
    
    sigma = np.linspace(0,0.02)
    
    adjustment_sigma = pd.DataFrame(data = vasicek_convexity_adjustment(r0, theta, kappa, sigma, t, T0, T1)*100*100,
                                   index = sigma*100)
    
    
    
    r0, theta, kappa, sigma = [0.06, 0.08, 0.86, 0.01]
    
    T0 = np.linspace(0,5)
    T1 = T0 + 0.25
    
    tmp = [vasicek_convexity_adjustment(r0, theta, kappa, sigma, t, T0[i], T1[i])*100*100 for i in range(len(T0))]
    adjustment_T0 = pd.DataFrame(data = tmp, index = T0)

    fig, axes = plt.subplots(1,2)

    axes[0].plot(adjustment_sigma)
    axes[0].set_xlabel('Volatility / %')
    
    axes[1].plot(adjustment_T0)
    axes[1].set_xlabel('Time to Maturity T0 - t')
    
    [x.set_ylabel('convexity adjusment / bps') for x in axes]
    [x.grid() for x in axes]
    fig.tight_layout()
    plt.show()
        
        
    r0, theta, kappa, sigma = [0.06, 0.08, 0.86, 0.01]
    t = 0.
    T0 = 1
    T1 = T0 + 0.25
    vasicek_convexity_adjustment(r0, theta, kappa, sigma, t, T0, T1)*100*100
    
