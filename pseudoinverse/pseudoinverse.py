#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:39:15 2018

@author: daniel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dateutil.relativedelta

def pseudoinverse(today, dates, quotes, types, rates = False, months = 1):
    if rates != False:
        data = pd.DataFrame({'Date': dates, 'Source': types, 
                             'Quote': quotes, 'Rate': quotes})
    else:
        data = pd.DataFrame({'Date': dates, 'Source': types, 
                             'Quote': quotes})

    C = pd.DataFrame()
    p = np.zeros(len(data))

    row = data.iloc[1,]

    for row in data.iterrows():
        i = row[0]
        
        row = row[1]
        typ = row.Source
        quote = row.Quote
        
        if rates != '':
            rate = row.Rate
        else:
            if typ == 'F':
                data.iloc[i,3] = 1 - quote/100
            else:
                data.iloc[i,3] = quote/100
    
            rate = data.iloc[i, 3]
            
        date = row.Date
        
        if typ == 'L':
            C_tmp = pd.DataFrame(data = [1. + (date-today).days/360.*rate], 
                                         columns = [date], index = [i])
            C = pd.concat([C, C_tmp], axis=1, join='outer')
            p[i] = 1.
            
        if typ == 'F':
            prev_date = date - dateutil.relativedelta.relativedelta(months=months)
            delta = (date-prev_date).days/360.
            C_tmp = pd.DataFrame({prev_date: -1., date: 1. + delta*rate}, index = [i])
            C = pd.concat([C, C_tmp], axis=0, join='outer').drop_duplicates().reset_index(drop=True)
            p[i] = 0.
    
        if typ == 'S':
            
            cf_dates = [date]
            
            while True:
                date = cf_dates[-1]
                date_tmp = pd.datetime(date.year - 1, date.month, date.day)
                if date_tmp < today:
                    break
                else:
                    cf_dates.extend([date_tmp])
            
            if len(cf_dates) == 1:
                cf_dates.extend([today])
            
            cf_dates = sorted(cf_dates)
            deltas = [x.days/360 for x in np.diff(cf_dates)]
            
            if cf_dates[0] > today:
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
    today = [today]
    today.extend(dates)
    
    C = C.as_matrix()
    
    deltas = np.diff(today)
    deltas = [x.days/360. for x in deltas]
    
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

if __name__ == '__main__':
    
    
    C = pd.read_excel('bootstrap.xls','pseudoinverse', skiprows = 27, usecols = [3 + x for x in range(39)])
    p = pd.read_excel('bootstrap.xls','pseudoinverse', skiprows = 27, usecols = [0])
    
    today = [pd.Timestamp('10/03/2012')]
    dates = list(C)
    today.extend(dates)
    
    C = C.as_matrix()
    p = np.array([float(x) for x in p.P])
    
    deltas = np.diff(today)
    deltas = [x.days/360. for x in deltas]
    
    vec = np.zeros(len(deltas))
    vec[0] = 1.
    vec[1:] = 0.
    
    W = np.diag(1./np.sqrt(deltas))
    M = np.diag([1]*len(deltas))
    
    for i in range(len(M)-1):
        M[i+1,i] = -1
        
    Mm1 = np.linalg.inv(M)
    Wm1 = np.linalg.inv(W)
    
    A = np.dot(C , np.dot(Mm1 , Wm1))
    A_m = np.dot(A.transpose(), np.linalg.inv(np.dot(A , A.transpose())))
    
    delta = np.dot(A_m, (p - np.dot(C , np.ones(len(deltas)) )))
    
    plt.plot(delta)
    
    d_tmp = np.dot(Wm1, delta) + vec
    d = np.dot(Mm1, d_tmp)
    
    plt.plot(d)
    
    answer = (dates[-1]-dates[-2]).days/360.*(d[-2]/d[-1]-1)
    print('The answer to the question is {:.2%}'.format(answer))



