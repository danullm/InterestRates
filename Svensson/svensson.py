#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:45:53 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def svensson_yields(beta0, beta1, beta2, beta3, tau1, tau2, T):
    T1 = T/tau1
    T2 = T/tau2
    tmp = beta0
    tmp += beta1 * (1-np.exp(-T1))/T1
    tmp += beta2 * ( (1-np.exp(-T1))/T1 - np.exp(-T1) )
    tmp += beta3 * ( (1-np.exp(-T2))/T2 - np.exp(-T2) )
    
    return(tmp)

def svensson_forwards(beta0, beta1, beta2, beta3, tau1, tau2, T):
    T1 = T/tau1
    T2 = T/tau2
    tmp = beta0
    tmp += beta1 * (np.exp(-T1))
    tmp += beta2 * T1 * np.exp(-T1)
    tmp += beta3 * T2 * np.exp(-T2)
    
    return(tmp)

def svensson_forwards_partial(beta0, beta1, beta2, beta3, tau1, tau2, T):
    T1 = T/tau1
    T2 = T/tau2
    tmp = -beta1/tau1*np.exp(-T1)
    tmp += beta2 / tau1 * (np.exp(-T1))
    tmp -= beta2 *T1 / tau1 * np.exp(-T1)
    tmp += beta3 / tau2 * (np.exp(-T2))
    tmp -= beta3 *T2 / tau2 * np.exp(-T2)
    return(tmp)


#------------------------------------------------------------------------------

if __name__ == '__main__':
    
#------------------------------------------------------------------------------
    #url = "https://www.bundesbank.de/cae/servlet/StatisticDownload?tsId=BBK01.SU0304&its_csvFormat=en&its_fileFormat=csv&mode=its"
    #eonia = pd.read_csv(url, index_col = 0, skiprows = 4, parse_dates = True, usecols = [0,1], skipfooter = 1)
    
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

#------------------------------------------------------------------------------

    t = np.arange(0,15,0.25)
    
    #yc = [svensson_yields(beta0, beta1, beta2, beta3, tau1, tau2, x) for x in t]
    fc = [svensson_forwards(beta0, beta1, beta2, beta3, tau1, tau2, x) for x in t]
    #dfc = [svensson_forwards_partial(beta0, beta1, beta2, beta3, tau1, tau2, x) for x in t]
    
    plt.plot(t, fc,  marker = '+')
    plt.ylabel('f(0,T) / %')
    plt.xlabel('Maturity T')
    plt.legend()
