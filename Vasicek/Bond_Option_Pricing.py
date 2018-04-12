#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:16:07 2018

@author: daniel
"""

from vasicek import *



def vasicek_bond_option(T0, T1, K, r0, theta, kappa, sigma, typ = 'call'):
    
    PT0 = np.exp(-vasicek_A(theta, kappa, sigma, T0, t = 0.) - vasicek_B(kappa, T0)*r0)
    PT1 = np.exp(-vasicek_A(theta, kappa, sigma, T1, t = 0.) - vasicek_B(kappa, T1)*r0)

    integral = sigma**2/kappa**2
    integral = integral * (np.exp(- kappa*T0) - np.exp(- kappa*T1))**2
    integral = integral * ( np.exp(2*kappa*T0) - 1 ) / (2*kappa)

    d1 = (np.log(PT1/K/PT0) + 0.5*integral)/np.sqrt(integral)
    d2 = (np.log(PT1/K/PT0) - 0.5*integral)/np.sqrt(integral)
    
    if typ == 'call':
        p = PT1*norm.cdf(d1)-K*PT0*norm.cdf(d2)
    else:
        p = K* PT0 * norm.cdf(-d2) - PT1 * norm.cdf(-d1)
        
    return(p)

if __name__ == '__main__':

    r0, theta, kappa, sigma = [0.06, 0.08, 0.86, 0.01]

    T0 = 0.25
    T1 = 0.5

    PT0 = np.exp(-vasicek_A(theta, kappa, sigma, T0, t = 0.) - vasicek_B(kappa, T0)*r0)
    PT1 = np.exp(-vasicek_A(theta, kappa, sigma, T1, t = 0.) - vasicek_B(kappa, T1)*r0)

    K = PT1/PT0
    
    vasicek_bond_option(T0, T1, K, r0, theta, kappa, sigma, typ = 'call')*10**4

