#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:15:01 2018

@author: daniel
"""

import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt

c = calendar.Calendar(firstweekday=calendar.SUNDAY)

dates = ["02/10/2012", "05/11/2012", "03/01/2013", "20/03/2013", "19/06/2013",
         "18/09/2013", "18/12/2013", "19/03/2014", "03/10/2014", "05/10/2015",
         "03/10/2016", "03/10/2017", "03/10/2019", "03/10/2022", "04/10/2027",
         "04/10/2032", "03/10/2042"]
dates = [pd.datetime.strptime(x, "%d/%m/%Y") for x in dates]

types = ['L']*3 + ['F'] * 5 + ['S'] * 9

quotes = [.15, .21, .36, 99.68, 99.67, 99.65, 99.64, 99.62, 
          .36, .43, .56, .75, 1.17, 1.68, 2.19, 2.40, 2.58]

data = pd.DataFrame({'Date': dates, 'Source': types, 'Quote': quotes})

data['Rate'] = quotes

today = pd.datetime.strptime("01/10/2012", "%d/%m/%Y")

C = pd.DataFrame()
p = np.zeros(len(quotes))

for row in data.iterrows():
    i = row[0]
    
    row = row[1]
    typ = row.Source
    quote = row.Quote
    
    #print(str(i) + " --/-- " + typ)
    
    if typ == 'F':
        data.iloc[i,3] = 1 - quote/100
    else:
        data.iloc[i,3] = quote/100
        
    rate = data.iloc[i, 3]
    date = row.Date
    
    print(rate)
    
    if typ == 'L':
        C_tmp = pd.DataFrame(data = [1. + (date-today).days/360.*rate], columns = [date], index = [i])
        C = pd.concat([C, C_tmp], axis=1, join='outer')
        p[i] = 1.
        
    if typ == 'F':
        prev_date = date - pd.Timedelta(weeks = 4*3)
        month = prev_date.month
        year = prev_date.year
        monthcal = c.monthdatescalendar(year,month)
        prev_date = [day for week in monthcal for day in week if \
                day.weekday() == calendar.WEDNESDAY and \
                day.month == month][2]
        
        prev_date = pd.datetime(year, month, prev_date.day)
        
        delta = (date-prev_date).days/360.
        
        C_tmp = pd.DataFrame({prev_date: -1., date: 1. + delta*rate}, index = [i])
        C = pd.concat([C, C_tmp], axis=0, join='outer').drop_duplicates().reset_index(drop=True)
        p[i] = 0.

    if typ == 'S':
        
        date_tmp = pd.datetime(date.year, date.month, 1)
        month_range = calendar.monthrange(date_tmp.year, date_tmp.month)
        
        date_corrected = pd.datetime(date_tmp.year, date_tmp.month, 1)
        delta = (calendar.MONDAY - month_range[0]) % 7
        monday = date_corrected + pd.Timedelta(days = delta)
        
        date = pd.datetime(monday.year, monday.month, monday.day)
        
        cf_dates = [date]
        while True:
            date_tmp = pd.datetime(date_tmp.year - 1, date_tmp.month, 1)
            if date_tmp < today:
                break
            else:
                month_range = calendar.monthrange(date_tmp.year, date_tmp.month)
                date_corrected = pd.datetime(date_tmp.year, date_tmp.month, 1)
                delta = (calendar.MONDAY - month_range[0]) % 7
                
                monday = date_corrected + pd.Timedelta(days = delta)

                prev_date = pd.datetime(monday.year, monday.month, monday.day)
                cf_dates.extend([prev_date])
        
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

libors = [(1/x[1]-1)*360./(x[0]-today[0]).days for x in zip(dates,d)]
yields = [-np.log(x[1])*360./(x[0]-today[0]).days for x in zip(dates,d)]


d_long = np.append(np.array(1), d)

deltas = [x.days/360 for x in np.diff(today)]

d0 = np.delete(d_long.copy(), -1)
d1 = np.delete(d_long.copy(), 0)

forwards = (d0/d1 - 1)/np.array(deltas)

#plt.plot(dates, d)
#
plt.plot(dates, libors, label = 'libors')
plt.plot(dates, yields, label = 'yields')
plt.plot(dates, forwards, label = 'forwards')
plt.grid()
plt.legend()

answer = (dates[-1]-dates[-2]).days/360.*(d[-2]/d[-1]-1)
print('The answer to the question is {:.2%}'.format(answer))
