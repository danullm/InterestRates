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

if __name__ == '__main__':
      
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
    yc = [svensson_yields(beta0, beta1, beta2, beta3, tau1, tau2, x)/100 for x in t]
    fc = [svensson_forwards(beta0, beta1, beta2, beta3, tau1, tau2, x)/100 for x in t]
    
    plt.scatter(t, fc,  marker = '+')
    plt.ylabel('f(0,T) / %')
    plt.xlabel('Maturity T')
    plt.legend()
    plt.show()
    