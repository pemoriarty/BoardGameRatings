#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:27:22 2018

@author: pamela
@manipulate and plot non-categorial data
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

#to_drop = [compiled_info[game]['id'] for game in range(len(compiled_info)) if compiled_info[game]['complexity'] == 0]

ages = [compiled_info[game]['age'] for game in range(len(compiled_info))]
ages = list(map(float, ages))
complexities = list(map(float, [compiled_info[game]['complexity'] for game in range(len(compiled_info))]))

nmech = [len(compiled_info[game]['mechanics']) for game in range(len(compiled_info))]
nmech = list(map(float, nmech))

nplayerrange = [int(compiled_info[game]['maxplayers']) - int(compiled_info[game]['minplayers']) for game in range(len(compiled_info))]

is_strategy = ['Strategy Games' in compiled_info[game]['subdomains'] for game in range(len(compiled_info))]
is_party = ['Party Game' in compiled_info[game]['categories'] for game in range(len(compiled_info))]
strat_bool = np.array(is_strategy) * 1
party_bool = np.array(is_party) * 1

all_vars = pd.DataFrame()
all_vars['ages'] = ages
all_vars['nmech'] = nmech
all_vars['strategy'] = strat_bool
all_vars['party'] = party_bool
all_vars['complexity'] = complexities
all_vars['nplayerrange'] = nplayerrange

all_no0 = all_vars[all_vars['complexity'] != 0 ]

plt.figure()
plt.hist((test['ages']), 10)

plt.figure()
plt.hist((test['nmech']), 10)#not normal!

plt.hist(np.log(test['complexity']), 10)#not normal

%matplotlib qt
plt.figure()
plt.scatter(ages, complexities)

plt.figure()
plt.scatter(ages, nmech)

plt.figure()
plt.scatter(nmech, complexities)

Xvars = pd.DataFrame()
Xvars['ages'] = all_no0['ages']
Xvars['nmech'] = all_no0['nmech']
#Xvars['strategy'] = all_no0['strategy']
Xvars['party'] = all_no0['party']
#Xvars['nplayerrange'] = all_no0['nplayerrange']


X_train, X_test, y_train, y_test = train_test_split(Xvars, np.log(all_no0['complexity']), test_size = 0.2)
print(X_train.shape)

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
model.coef_
model.intercept_
model.score(X_test, y_test)

predictions = lm.predict(X_test)
plt.figure()
plt.scatter(np.exp(y_test), np.exp(predictions))
x = range(4)
y = range(4)
plt.plot(x, y)

#####try a gamma model
gamma_model = sm.GLM(all_no0['complexity'], Xvars.astype(float), family=sm.families.Gamma())
gamma_results = gamma_model.fit()
print(gamma_results.summary())

nobs = gamma_results.nobs
y = all_no0['complexity']
yhat = np.log(gamma_results.mu)

from statsmodels.graphics.api import abline_plot
fig, ax = plt.subplots()
ax.scatter(y, yhat)
line_fit = sm.OLS(y, yhat).fit()
abline_plot(model_results=line_fit, ax=ax)

#plot predicted vs observed
ax.set_title('Model Fit Plot')
ax.set_ylabel('Observed values')
ax.set_xlabel('Fitted values')

fig, ax =e plt.subplots()

#plot residuals
fig, ax = plt.subplots()
ax.scatter(yhat, gamma_results.resid_pearson)
ax.hlines(0, 0, 1)
ax.set_xlim(0, 1)
ax.set_title('Residual Dependence Plot')
ax.set_ylabel('Pearson Residuals')
ax.set_xlabel('Fitted values')
