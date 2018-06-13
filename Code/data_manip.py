#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:24:46 2018

@author: pamela
"""

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.special
#import statsmodels.api as sm
#import statsmodels.genmod as
#import statsmodels as sm
#from sklearn import datasets, linear_model
#from sklearn.model_selection import train_test_split

#file_name = "/media/pamela/Stuff/compiled_info"
file_name = '/home/pamela/Documents/xmls_parsed'
fileObject = open(file_name, 'rb')
with open(file_name, 'rb') as f:
    full_info = pickle.load(f)
fileObject.close()

#full_info = list(compiled_info)
#remove games without a complexity rating or with fewer than 10 people having rated it
red_info = list(full_info)

idx_to_del = []
for i in range(len(red_info)):
    try:
        if int(red_info[i]['num_comp']) < 20:
            idx_to_del.append(i)
    except TypeError:
        idx_to_del.append(i)

for i in range(len(idx_to_del)-1, 0, -1):
    del red_info[idx_to_del[i]]

#pull out the variables I'm using
ages = [red_info[game]['age'] for game in range(len(red_info))]
#ages = [np.nan if v is None else v for v in ages]
complexities = [red_info[game]['complexity'] for game in range(len(red_info))]
nmech = [len(red_info[game]['mechanics']) for game in range(len(red_info))]
is_strategy = ['Strategy Games' in red_info[game]['subdomains'] for game in range(len(red_info))]
is_party = ['Party Game' in red_info[game]['categories'] for game in range(len(red_info))]
strat_bool = np.array(is_strategy) * 1
party_bool = np.array(is_party) * 1

df_info = pd.DataFrame()
df_info['ages'] = ages
df_info['complexities'] = complexities
df_info['nmech'] = nmech
df_info['is_strategy'] = strat_bool
df_info['is_party'] = party_bool


idx_to_del = []
for i in range(len(df_info)):
    try:
        if df_info['complexities'][i] == 0 or df_info['ages'][i] == 0:
            idx_to_del.append(i)
    except NameError:
        idx_to_del.append(i)

for i in range(len(idx_to_del)-1, 0, -1):
    del df_info[idx_to_del[i]]





##make a dataframe of the cleaned and processed variables I'm using
#all_vars = pd.DataFrame()
#all_vars['ages'] = ages
#all_vars['nmech'] = nmech
#all_vars['strategy'] = strat_bool
#all_vars['party'] = party_bool
#all_vars['complexity'] = complexities
##all_vars['nplayerrange'] = nplayerrange


all_no0 = all_vars[all_vars['complexity'] != 0]

bounded_y = (df_info['complexities'] - 1)/4
bounded_y[bounded_y == 0]
bounded_y[bounded_y == 1]
bounded_y[bounded_y == 0] = .001
bounded_y[bounded_y == 1] = 0.99
continuous_y = scipy.special.logit(bounded_y)
df_info['response'] = continuous_y
#continuous_y[np.isnan(continuous_y)]

plt.figure()
plt.hist(continuous_y)#approximatley normal!

##################################
#look at is_party
##################################
plt.figure()
plt.hist(all_vars['party'])

party_dummies = pd.get_dummies(all_vars['party'])

#################################
#look at ages
##################################
plt.figure()
plt.hist(ages)

all_vars['ages'].max()
#all_vars[all_vars['ages'] == 0].index()
#ages[np.isnan(ages)]

plt.scatter(ages_tmp, continuous_y == 1.8421)
continuous_y)
#feature engineering: turn ages into 3 categories: 0, =< 12, > 12
ages_tmp = np.asarray(all_vars['ages'])
ages_tmp = np.asarray(ages)

all(ages_tmp == 0)

#idx_0 = []
##idx_0 = [idx_0.append(i) for i in range(len(ages_tmp)) if ages_tmp[i] == 0]
#for i in range(len(ages_tmp)):
#    if ages_tmp[i] == 0:
#        idx_0.append(i)
#        
#idx_13 = []
#for i in range(len(ages_tmp)):
#    if ages_tmp[i] > 12:
#        idx_13.append(i)
#        
#idx_12 = []
#for i in range(len(ages_tmp)):
#    if ages_tmp[i] > 0 and ages_tmp[i] <13:
#        idx_12.append(i)
        
#ages_tmp[idx_0] = 0
#ages_tmp[idx_12] = 1
#ages_tmp[idx_13] = 2

#age_dummies = pd.get_dummies(ages_tmp)

###########################################
#number of mechanics
###########################################
plt.hist(all_vars['nmech'])
