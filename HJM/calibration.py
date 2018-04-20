#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:53:16 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize as sop

def pseudoinverse(dates, rates, types):
    data = pd.DataFrame({'Date': dates, 'Source': types, 'Rates': rates})
    
    C = pd.DataFrame()
    p = np.zeros(len(data))
    
    for row in data.iterrows():
        i = row[0]
            
        row = row[1]
        typ = row.Source
        rate = row.Rates
        date = row.Date
        
        if typ == 'L':
            C_tmp = pd.DataFrame(data = [1. + date*rate], columns = [date], index = [i])
            C = pd.concat([C, C_tmp], axis=1, join='outer')
            p[i] = 1.
        
        if typ == 'S':
        
            cf_dates = [date]
                    
            while True:
                date = cf_dates[-1]
                date_tmp = date - 1
                if date_tmp < 0:
                    break
                else:
                    cf_dates.extend([date_tmp])
            
            cf_dates = sorted(cf_dates)
            deltas = np.diff(cf_dates)
            
            if cf_dates[0] > 0:
                cfs = [-1.]
                cfs.extend([x*rate for x in deltas])
                cfs[-1] += 1
                p[i] = 0.
            else:
                p[i] = 1.
                cfs = [x*rate for x in deltas]
                cfs[-1] += 1
                cf_dates.pop(0)
                
            C_tmp = pd.DataFrame(data = cfs, index = cf_dates).transpose()
            C = pd.concat([C, C_tmp], axis=0, join='outer').drop_duplicates().reset_index(drop=True)
        
    C = C.reindex_axis(sorted(C.columns), axis=1)
    C = C.fillna(0)
    
    p = np.array(p)
    
    dates = list(C)
    today = [0]
    today.extend(dates)
    
    C = C.as_matrix()
    
    deltas = np.diff(today)
    vec = np.zeros(len(deltas))
    vec[0] = 1.
    vec[1:] = 0.
    vec2 = np.zeros(len(deltas))
    vec2[:] = 1.
    np.ones(len(deltas))
    W = np.diag(1./np.sqrt(deltas))
    M = np.diag([1]*len(deltas))
    
    for i in range(len(M)-1):
        M[i+1,i] = -1
        
    Mm1 = np.linalg.inv(M)
    Wm1 = np.linalg.inv(W)
    A = np.dot(C , np.dot(Mm1 , Wm1))
    A_m = np.dot(A.transpose(), np.linalg.inv(np.dot(A , A.transpose())))
    delta = np.dot(A_m, (p - np.dot(C , np.ones(len(deltas)) )))
    d_tmp = np.dot(Wm1, delta) + vec
    d = np.dot(Mm1, d_tmp)

    return(pd.DataFrame(data = d, index = dates))


def forward_curve(discount):
    dates = discount.index
    delta = np.diff(dates)[0]
    P0 = float(discount.iloc[0,])
    discount_tmp = discount.iloc[1:,]
    sum_df = discount_tmp.cumsum() * delta
    sum_df = sum_df[1:]
    diff = [P0 - float(x[1]) for x in discount_tmp.iterrows()]
    diff = pd.DataFrame(diff, index = sum_df.index)
    return(diff/sum_df)
    
    
def bond_vola_integral(betas, volas, times):
    ret = 0
    for i in range(len(betas)):
        tmp = volas[i]**2/betas[i]**2 
        tmp *= ( np.exp( -betas[i]*times[0] ) - np.exp( -betas[i]*times[1] ) )**2
        tmp *= ( np.exp( 2*betas[i]*times[0] ) - 1 )/2/betas[i]
        ret += tmp
    return(ret)
    
    
def cpl_Gauss_HJM(betas, volas, times, k, discount):
    T1 = times[1]
    T0 = times[0]
    delta = T1-T0 
    d1, d2 = cpl_Gauss_HJM_d(betas, volas, times, k, discount)
    tmp = 1+delta*k
    tmp2 = discount.loc[T0]*norm.cdf(-d2) - tmp * discount.loc[T1]*norm.cdf(-d1)
    return(tmp2)


def cpl_Gauss_HJM_d(betas, volas, times, k, discount):
    T1 = times[1]
    T0 = times[0]
    delta = T1-T0
    tmp1 = np.log(discount.loc[T1]/discount.loc[T0]*(1+delta*k))
    tmp2 = bond_vola_integral(betas, volas, times)
    return([(tmp1+0.5*tmp2)/np.sqrt(tmp2), (tmp1-0.5*tmp2)/np.sqrt(tmp2)])


def cp_Gauss_HJM(T, k, betas, volas, discount):
    val = 0
    for i in range(len(T)-1):
        times = [T[i], T[i+1]]
        val += cpl_Gauss_HJM(betas, volas, times, k, discount)
    return(val)

def error_function(p0):
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
    
    global i, min_RMSE, discount
    
    v1, v2, beta1, beta2 = p0
    
    volas = [v1, v2]
    betas = [beta1, beta2]
    
    se = []
    for t, x in calibration_data.iterrows():
        k = x[2]
        fy = x[1]
        Tmax = x[0]
        T = np.arange(0.5, Tmax+0.5, 0.5)
        model_value = cp_Gauss_HJM(T, k, betas, volas, discount)
        se.append((model_value - fy) ** 2)
        
    RMSE = float(np.sqrt(sum(se) / len(se)))
    min_RMSE = min(min_RMSE, RMSE)
    
    if i % 50 == 0:
       print('%4d |' % i, '%2.3f, %2.3f, %2.3f, %2.3f' % tuple(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE))
    i += 1
    
    return RMSE


if __name__ == '__main__':
    
    i = 0           # counter initialization
    min_RMSE = 100  # minimal RMSE initialization
    
    dates = [0.5, 1,2,3,4,5,6,7,8,9,10,15,20,30]
    rates = [0.3430, 0.4420, 0.6260, 0.8630, 1.1191, 
             1.3650, 1.5750, 1.7574, 1.9184, 2.0630,
             2.1905, 2.5990, 2.7135, 2.7135]
    rates = [x/100 for x in rates]
    types = ['L'] + ['S']*13
    
    cap_maturity = [x for x in np.arange(1,11,1)] + [15,20,30]
    cap_maturity = [float(x) for x in cap_maturity]
    cap_price = [0.0012, 0.0046, 0.0092, 0.0148, 0.0210, 0.0278, 0.0349,
                 0.0417, 0.0490, 0.0565, 0.0904, 0.1196, 0.1686]
    
    black_iv = [170.52, 113.62, 76.52, 54.54, 41.36, 34.58,
                30.46, 27.10, 25.02, 23.67, 19.87, 19.38, 19.31]
    black_iv = [x/100 for x in black_iv]
    
    normal_iv = [86.81, 76.58, 70.92, 67.17, 63.86, 62.10, 60.79, 58.67,
                 57.49, 56.86, 53.46, 54.80, 56.79]
    normal_iv = [x/100/100 for x in normal_iv]

    
    calibration_data = pd.DataFrame({'Maturity': cap_maturity, 'Price': cap_price})
    
    discount = pseudoinverse(dates, rates, types)

    tmp = pd.DataFrame(index = np.arange(0.5,30.5,0.5))
    
    discount = pd.concat([tmp, discount], axis = 1)
    discount = discount.interpolate('index', axis = 0)

    forward_rates = forward_curve(discount)


    calibration_data['ATM_strikes'] = forward_rates[forward_rates.index.isin(calibration_data.Maturity.values)].values

    p0 = sop.brute(error_function,
               ((0, 0.1, 0.05),  #v1,
                (0, 0.1, 0.05),  #v2,
                (0, 2., 0.1),           #beta1, 
                (0, 2., 0.1)),          #beta2
                finish=None)
               
    opt = sop.fmin(error_function, p0, maxiter=5000, 
               maxfun=750, xtol=0.000001, ftol=0.000001)

