#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:27:22 2018

@author: pamela
@manipulate and plot non-categorial data
"""
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
import statsmodels.genmod as 
import statsmodels as sm
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

#file_name = "/media/pamela/Stuff/compiled_info"
file_name = '/home/pamela/Documents/xmls_parsed.pickle'
fileObject = open(file_name,'rb') 
with open(file_name, 'rb') as f:
    full_info = pickle.load(f)
fileObject.close()   


#to_drop = [compiled_info[game]['id'] for game in range(len(compiled_info)) if compiled_info[game]['complexity'] == 0]
#del compiled_info[43610]

red_info = list(full_info)

idx_to_del = []
for i in range(len(red_info)):
    if red_info[i]['complexity'] == '0' or red_info[i]['age'] == '':
        idx_to_del.append(i)

    
for i in range(len(idx_to_del)-1,0, -1):       
    del red_info[idx_to_del[i]]

    
idx_to_del = []
for i in range(len(red_info)):
    if red_info[i]['users_rated'] < '10':
        idx_to_del.append(i)
for i in range(len(idx_to_del)-1,0, -1):       
    del red_info[idx_to_del[i]]
    

ages = [red_info[game]['age'] for game in range(len(red_info))]
ages = list(map(float, ages))
complexities =  [red_info[game]['complexity'] for game in range(len(red_info))]
complexities = list(map(float, complexities))

for i in range(len(ages)):
    try: 
        ages[i] = float(ages[i])
    except ValueError:
        print(i)

nmech = [len(red_info[game]['mechanics']) for game in range(len(red_info))]
nmech = list(map(float, nmech))

#nplayerrange = [int(red_info[game]['maxplayers']) - int(red_info[game]['minplayers']) for game in range(len(compiled_info))]

is_strategy = ['Strategy Games' in red_info[game]['subdomains'] for game in range(len(red_info))]
is_party = ['Party Game' in red_info[game]['categories'] for game in range(len(red_info))]
strat_bool = np.array(is_strategy) * 1
party_bool = np.array(is_party) * 1

all_vars = pd.DataFrame()
all_vars['ages'] = ages
all_vars['nmech'] = nmech
all_vars['strategy'] = strat_bool
all_vars['party'] = party_bool
all_vars['complexity'] = complexities
#all_vars['nplayerrange'] = nplayerrange


all_no0 = all_vars[all_vars['complexity'] != 0 ]
all_no0['std_complex'] =  all_no0['complexity']/5.0001

plt.figure()
plt.hist((test['ages']), 10)

plt.figure()
plt.hist((test['nmech']), 10)#not normal!

plt.hist(np.log(test['complexity']), 10)#not normal
plt.figure()
plt.hist(all_no0['std_complex'])


test = scipy.special.logit(all_no0['std_complex'])
test[np.isnan(test)]
plt.figure()
plt.hist(test, 20)



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


X_train, X_test, y_train, y_test = train_test_split(Xvars, np.log(all_no0['std_comp']), test_size = 0.2)
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

######
#gausssian model with logit link
logit_link  = sm.GLM(test, sm.add_constant(Xvars.astype(float)), family = sm.families.Gaussian(link = sm.families.links.identity))
logit_results = logit_link.fit()
print(logit_results.summary())

fits = logit_results.fittedvalues
#pred = logit_results.p
res = logit_results.resid_pearson

fig, ax = plt.subplots()
ax.scatter(fits, res)
ax.hlines(0, 0, 1)

fig, ax = plt.subplots()
ax.scatter(scipy.special.expit(test)*5, scipy.special.expit(fits)*5)
ax.scatter((test), (fits))

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
