#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:45:31 2018

@author: daniel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
###############################################################################

data = pd.read_csv('SwissGovYields.csv', index_col = 0, parse_dates = True)

data = data/100

yields = data["2010-06"]

Year = [2, 3, 4, 5]
CF = [80, 70, 150, 40]
yields = yields.iloc[0, 0:4]

portfolio = pd.DataFrame({'Year': Year, 'CF': CF, 'Yield': yields})

###############################################################################
###############################################################################

changes = data.diff().dropna()

cov_mat = np.cov(changes, rowvar = False)

lambdas, A = np.linalg.eig(cov_mat)

Ainv = np.linalg.inv(A)

firt_two = sum(lambdas[0:2])/(sum(lambdas))

print('The answer to the question is {:.2%}'.format(firt_two ))
print('--------------------------------------------------------')
print(' ')

###############################################################################
###############################################################################

mean_changes = np.array(changes.mean(axis = 0))

# demeaned changes in euklidean space
X = changes.sub(mean_changes)

# demeand changes in eigen space
Y = np.dot(X, A)

Y_reduced = Y[:,0:2]
A_reduced = A[:,0:2]
A_reduced_transposed = A_reduced.transpose()

X1 = np.dot(Y_reduced, A_reduced_transposed)

dy = X1 + mean_changes

pca = dy[:,0:4]

dt = 1./12

partial_t = sum(portfolio.Yield.values * portfolio.CF.values * np.exp(- portfolio.Year.values * portfolio.Yield.values))
partial_y = - portfolio.CF.values * portfolio.Year.values * np.exp(- portfolio.Year.values * portfolio.Yield.values)

sum(portfolio.CF.values * np.exp(-portfolio.Yield.values * portfolio.Year.values ))

dV = partial_t * dt + np.dot(pca,partial_y)

answer1 = dV.std()


real_changes = np.dot(changes.iloc[:,0:4] ,partial_y)
real_changes += partial_t * dt


print('The answer to the question is {:.2f}'.format(answer1))
print('--------------------------------------------------------')
print(' ')

plt.plot(dV, label = 'pca')
plt.plot(real_changes, label = 'real')
plt.grid()
plt.legend()
plt.tight_layout()
