#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:16:07 2018

@author: daniel
"""

from vasicek import *

def vasicek_bond_option(T0, T1, K, r0, theta, kappa, sigma, typ = 'call'):
    if T0 == 0:
        PT0 = 1.
    else:
        PT0 = vasicek_discount_curve(r0, theta, kappa, sigma, T = T0, N = 2).loc[T0,0]
        
    PT1 = vasicek_discount_curve(r0, theta, kappa, sigma, T = T1, N = 2).loc[T1,0]

    integral = sigma**2/kappa**2*(np.exp(-kappa*T0)-np.exp(-kappa*T1))**2*(np.exp(2*kappa*T0)-1)/2/kappa

    d1 = (np.log(PT0/K/PT1) + 0.5*integral)/np.sqrt(integral)
    d2 = (np.log(PT0/K/PT1) - 0.5*integral)/np.sqrt(integral)
    if typ == 'call':
        p = PT1*norm.cdf(d1)-K*PT0*norm.cdf(d2)
    else:
        p = K*PT0*norm.cdf(-d2) - PT1*norm.cdf(-d1)
        
    p = p.values
    return(p)

if __name__ == '__main__':

    r0, theta, kappa, sigma = [0.06, 0.08, 0.86, 0.01]

    P = vasicek_discount_curve(r0, theta, kappa, sigma, T = 10, N = 41)

    Ks = P.values[1:]/P.values[:-1]

    p = []

    for i in range(1,40):
                
        T0 = P.index[i]
        T1 = P.index[i+1]
        K = Ks[i][0]
        
        p.append(vasicek_bond_option(T0 , T1, K, r0, theta, kappa, sigma, typ = 'call'))

    plt.plot(p)
