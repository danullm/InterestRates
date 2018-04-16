#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:39:06 2018

@author: daniel
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize as sop
import sys
sys.path.insert(0, '/home/daniel/Seafile/Dani/Python/InterestRates/Svensson/')

from svensson import *
from vasicek import *

#------------------------------------------------------------------------------

def Vaiscek_error_function(p0):
    ''' Error Function for parameter calibration in the Vasicek Model

    Parameters
    ==========
    r0: float
        current short rate
    theta: float
        Vasicek long term rate
    kappa: float
        mean reversion speed
    sigma: float
        volatility of short rate

    Returns
    =======
    RMSE: float
        root mean squared error
    '''
    
    global i, min_RMSE
    
    r0, theta, kappa, sigma = p0
    
    if sigma < 0.0 or kappa < 0.0:
        return 500.0
    
    se = []
    for t, x in forward_curve.iterrows():
        fy = x[0]
        model_value = vasicek_forward_curve(r0, theta, kappa, sigma, T = t, N = 2).loc[t,0]
        se.append((model_value - fy) ** 2)
        
    RMSE = np.sqrt(sum(se) / len(se))
    min_RMSE = min(min_RMSE, RMSE)
    
    if i % 50 == 0:
       print('%4d |' % i, '%2.3f, %2.3f, %2.3f, %2.3f' % tuple(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE))
    i += 1
    
    return RMSE


#------------------------------------------------------------------------------

if __name__ == '__main__':
    
    i = 0           # counter initialization
    min_RMSE = 100  # minimal RMSE initialization
    
    
    plot = True
    
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
    [beta0, beta1, beta2, beta3] = [x / 100 for x in [beta0, beta1, beta2, beta3]] 
    
    t = np.arange(1./12,10,1./12)
    yc = [svensson_yields(beta0, beta1, beta2, beta3, tau1, tau2, x) for x in t]
    fc = [svensson_forwards(beta0, beta1, beta2, beta3, tau1, tau2, x) for x in t]
    
    forward_curve = pd.DataFrame(data = fc, index = t)
    
    if plot == True:
        plt.scatter(t, fc,  marker = '+')
        plt.ylabel('f(0,T) / %')
        plt.xlabel('Maturity T')
        plt.legend()
        plt.show()
    
    p0 = sop.brute(Vaiscek_error_function,
               ((-0.1, 0.1, 0.05),  #r0, 
                (-0.1, 0.1, 0.05),  #theta,  
                (0.1, 2., 0.05),      #kappa, 
                (0.01, 0.30, 0.05)),    #sigma
                finish=None)
               
    opt = sop.fmin(Vaiscek_error_function, p0, maxiter=5000, 
               maxfun=750, xtol=0.000001, ftol=0.000001)

    vasicek_fit = vasicek_forward_curve(opt[0], opt[1], opt[2], opt[3], T = max(t), N = 50)
    
    plt.plot(vasicek_fit, label = 'Model')
    plt.scatter(t, fc, label = 'Data', marker = '+', c = 'red')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    

