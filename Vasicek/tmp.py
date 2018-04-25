#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 13:24:23 2018

@author: daniel
"""

from vasicek import *

r0, theta, kappa, sigma = [0.06, 0.08, 0.86, 0.01]

tmp = sigma**2/2/kappa**2*( np.exp(-kappa) - 1 )**2

tmp*100*100


r0, theta, kappa, sigma = [0.08, 0.09, 0.86, 0.0148]

discount = vasicek_discount_curve(r0, theta, kappa, sigma, T = 30, N = 121)

cap_rate = float((discount.loc[0.25] - discount.loc[30])/0.25/(discount.sum()-discount.loc[0.25]))

