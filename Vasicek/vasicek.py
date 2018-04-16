#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:36:13 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

import sys
sys.path.insert(0, '/home/daniel/Seafile/Dani/Python/InterestRates/Svensson/')

#------------------------------------------------------------------------------

def vasicek_process(r0, theta, kappa, sigma, T = 1., N = 100, seed = 0):
    if seed != 0:
        np.random.seed(seed)

    dt = T/N
    rates = [r0]
    for i in range(N):
        dr = kappa*(theta-rates[-1])*dt + sigma*np.random.normal(size = 1, scale = np.sqrt(dt))
        rates.append(rates[-1] + dr)
    
    return(pd.DataFrame(data = rates, index = [x*dt for x in range(N+1)] ))

#------------------------------------------------------------------------------

def vasicek_mean(r0, theta, kappa, sigma, T = 1., N = 100):
    t = range(N+1)
    dt = T/N
    t = [x*dt for x in t]
    y = [np.exp(-kappa*x)*r0+theta*(1-np.exp(-kappa*x)) for x in t]
    y = np.array(y)
    return(pd.DataFrame(data = y, index = t))

#------------------------------------------------------------------------------

def vasicek_sd(r0, theta, kappa, sigma, T = 1., N = 100, alpha = 0.90):
    dt = T/N
    alpha = 1-alpha
    t = range(N+1)
    t = [x*dt for x in t]
 
    y = np.sqrt(np.array([sigma**2/2/kappa*(1-np.exp(-2*kappa*x)) for x in t]))

    means = vasicek_mean(r0, theta, kappa, sigma, T, N)
    lower = pd.DataFrame(data = y*norm.cdf(1-alpha/2), index = means.index)
    upper = means + lower
    lower = means - lower
        
    return(lower, upper)

#------------------------------------------------------------------------------

def vasicek_B(kappa, T, t = 0.):
    if t > 0:
        T = T-t
    return( (1-np.exp(-kappa*T))/kappa )

#------------------------------------------------------------------------------
    
T = 1
    
def vasicek_A(theta, kappa, sigma, T, t = 0.):
    if t > 0:
        T = T-t
       
    tmp = -(theta-sigma**2/2/kappa**2)*(vasicek_B(kappa, T, t) - T + t)
    tmp += sigma**2/4/kappa * vasicek_B(kappa, T, t)**2
    return(tmp)

#------------------------------------------------------------------------------

def vasicek_discount_curve(r0, theta, kappa, sigma, T = 10, N = 50):
    t = np.linspace(0,T,N)
    A = np.array([vasicek_A(theta, kappa, sigma, T) for T in t])
    B = np.array([vasicek_B(kappa, T) for T in t])
    
    tmp = np.exp(-A-B*r0)
    tmp = pd.DataFrame(data = tmp, index = t)
    
    return(tmp)

#------------------------------------------------------------------------------

def vasicek_yield_curve(r0, theta, kappa, sigma, T = 10, N = 50):
    discount_curve = vasicek_discount_curve(r0, theta, kappa, sigma, T, N)
    t = discount_curve.index
    
    y = [r0]
    for x in t[1:]:
        y.append(-np.log(discount_curve.loc[x,:].values[0])/x)
    
    tmp = pd.DataFrame(data = y, index = t)    

    return(tmp)
    
    
def vasicek_forward_curve(r0, theta, kappa, sigma, T = 10, N = 50):
    t = np.linspace(0,T,N)
    A = np.array([vasicek_A(theta, kappa, sigma, T) for T in t])
    B = np.array([vasicek_B(kappa, T) for T in t])
    
    tmp = kappa*theta*B
    tmp += r0*(1 - sigma**2/2*B**2 - kappa*B)
    
    tmp = pd.DataFrame(data = tmp, index = t)
    
    return(tmp)

    
#------------------------------------------------------------------------------

if __name__ == '__main__':

    r0, theta, kappa, sigma = [0.06, 0.08, 0.86, 0.01]
    
    T , N, mc = [25., 500, 5]
    
    plot = False
    
    rates = vasicek_process(r0, theta, kappa, sigma, T, N)
    for i in range(mc-1):
        rates = pd.concat([rates, vasicek_process(r0, theta, kappa, sigma, T, N)], axis = 1)
    
    means = vasicek_mean(r0, theta, kappa, sigma, T, N)
    lower, upper = vasicek_sd(r0, theta, kappa, sigma, T, N, alpha = .99)
    
    DFC = vasicek_discount_curve(r0, theta, kappa, sigma, T)    
    YC = vasicek_yield_curve(r0, theta, kappa, sigma, T)
    FC = vasicek_forward_curve(r0, theta, kappa, sigma, T)
    
    if plot == True:
        
        fig, axes = plt.subplots(3,1, sharex = True)
        
        axes[0].plot(rates, alpha = 0.25)
        axes[0].plot(means, c = 'red')
        axes[0].plot(lower, c = 'red', linestyle = '--')
        axes[0].plot(upper, c = 'red', linestyle = '--')
        axes[0].axhline(theta, c = 'black', linestyle = ':')
        
        axes[1].plot(DFC, label = 'Discount Curve')
        axes[1].legend()
        
        axes[2].plot(YC, label = 'Yield Curve')
        axes[2].plot(FC, label = 'Forward Curve')
        axes[2].legend()
        
        [x.grid() for x in axes]
        fig.tight_layout()
        plt.show()
        
#------------------------------------------------------------------------------
        
    names = {"BBK01.WZ9801": 'beta0',
             "BBK01.WZ9802": 'beta1', 
             "BBK01.WZ9803": 'beta2',
             "BBK01.WZ9805": 'beta3',
             "BBK01.WZ9804": 'tau1',
             "BBK01.WZ9806": 'tau2'}
    
    parameters = pd.DataFrame()
    
    for name in list(names.keys()):
        print(name)
        url = "https://www.bundesbank.de/cae/servlet/StatisticDownload?tsId="
        url += name
        url += "&its_csvFormat=en&its_fileFormat=csv&mode=its"
         
        tmp = pd.read_csv(url, index_col = 0, skiprows = 4, parse_dates = True, usecols = [0,1])
        tmp.columns = [names[name]]
         
        parameters = pd.concat([parameters, tmp], axis = 1)        


    parameters = parameters.reindex_axis(sorted(parameters.columns), axis = 1)    
    [beta0, beta1, beta2, beta3, tau1, tau2] = parameters.iloc[-1:].values[0]
    
    
    t = np.arange(1,10,1)
    yc = [svensson_yields(beta0, beta1, beta2, beta3, tau1, tau2, x) for x in t]
    fc = [svensson_forwards(beta0, beta1, beta2, beta3, tau1, tau2, x) for x in t]
    plt.scatter(t, fc,  marker = '+')
    plt.ylabel('f(0,T) / %')
    plt.xlabel('Maturity T')
    plt.legend()
    plt.show()