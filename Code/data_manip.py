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
file_name = "/media/pamela/Stuff/xmls_parsed"
file_name = "/home/pamela/Dropbox/xmls_parsed"
fileObject = open(file_name, 'rb')
with open(file_name, 'rb') as f:
    full_info = pickle.load(f)
fileObject.close()

#full_info = list(compiled_info)

#remove games without a complexity rating or with fewer than 20 people having rated it
red_info = list(full_info)

idx_to_del = []
for i in range(len(red_info)):
    try:
        if red_info[i]['num_comp'] < 20 or red_info[i]['complexity'] == 0:
            idx_to_del.append(i)
    except TypeError:
        idx_to_del.append(i)

for i in range(len(idx_to_del)-1, 0, -1):
    del red_info[idx_to_del[i]]

#move all variables to a dataframe
df_info = pd.DataFrame()    
for game in range(len(red_info)):
    df_tmp = pd.DataFrame.from_dict(red_info[game], orient = 'index')
    df_info = pd.concat([df_info, df_tmp], axis = 1)

df_info.index.rename('attribute', inplace = True)
df_info2 = df_info.transpose().copy()

pub_string = pd.Series()
for game in range(df_info2.shape[0]):
    pub_string.loc[game] = (''.join(np.asarray(df_info2['publisher'])[game]))

df_info2 = df_info2.assign(all_pub = (pub_string))#.Series().values)
df_info2.columns
#df_info2['std_comp'] = 
#test4.loc['categories']
#np.asarray(test4.loc['categories'])[0]
#test3.iloc[0]
#test4.T
#make variables for regression: nmech, is_party
nmech = [len(np.asarray(df_info2['mechanics'])[game]) for game in range(df_info2.shape[0])]
is_party = ['Party Game' in np.asarray(df_info2['categories'])[game] for game in range(df_info2.shape[0])]
party_bool = np.array(is_party) * 1
is_strategy = ['Strategy Games' in np.asarray(df_info2['subdomains'])[game] for game in range(df_info2.shape[0])]
strategy_bool = np.array(is_strategy) * 1
is_german = ['Germany' in np.asarray(df_info2['all_pub'])[game] for game in range(df_info2.shape[0])]
german_bool = np.array(is_german) * 1


sub_df = pd.DataFrame()
sub_df['id'] = df_info2['id']
sub_df['ages'] = df_info2['age']
sub_df['nmech'] = nmech
sub_df['is_party'] = party_bool
sub_df['complexity'] = df_info2['complexity']
sub_df['is_strategy'] = is_strategy
sub_df['nplayers'] = player_range

bounded_y = np.array((df_info2['complexity'] - 1)/4)
#bounded_y[bounded_y == 0]
#bounded_y[bounded_y == 1]
bounded_y[bounded_y == 0] = .001
bounded_y[bounded_y == 1] = 0.99

continuous_y = np.empty([len(bounded_y), 1])
for idx in range(len(bounded_y)):
    continuous_y[idx] = (scipy.special.logit(bounded_y[idx]))
    
#continuous_y = scipy.special.logit(np.asarray(bounded_y))

sub_df['response'] = continuous_y
sub_df.reset_index(drop = 0, inplace = True)


ages_0_idx = sub_df['ages'][sub_df['ages'] == 0].index.values

sub_df2 = sub_df.copy(deep = True)

for i in range(len(ages_0_idx)-1, -1, -1):
    #del sub_df2[ages_0_idx[i]]
    sub_df2.drop(ages_0_idx[i], inplace = True)


#remove nan's
    
    
    
#ages = np.asarray([red_info[game]['age'] for game in range(len(red_info))])

#ages = [np.nan if v is None else v for v in ages]
#complexities = [red_info[game]['complexity'] for game in range(len(red_info))]
#nmech = [len(red_info[game]['mechanics']) for game in range(len(red_info))]
#is_strategy = ['Strategy Games' in red_info[game]['subdomains'] for game in range(len(red_info))]
#is_party = ['Party Game' in red_info[game]['categories'] for game in range(len(red_info))]
#strat_bool = np.array(is_strategy) * 1
#party_bool = np.array(is_party) * 1
#
#df_info = pd.DataFrame()
#df_info['ages'] = ages
#df_info['complexities'] = complexities
#df_info['nmech'] = nmech
#df_info['is_strategy'] = strat_bool
#df_info['is_party'] = party_bool


#idx_to_del = []
#for i in range(len(df_info)):
#    try:
#        if df_info['complexities'][i] == 0 or df_info['ages'][i] == 0:
#            idx_to_del.append(i)
#    except NameError:
#        idx_to_del.append(i)
#
#for i in range(len(idx_to_del)-1, 0, -1):
#    del df_info[idx_to_del[i]]


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
plt.hist(sub_df2['ages'])

#all_vars[all_vars['ages'] == 0].index()
#ages[np.isnan(ages)]

plt.scatter(sub_df2['ages'], sub_df2['response'])

###########################################
#number of mechanics
###########################################
plt.hist(all_vars['nmech'])

######################################
#if published in Germany
plt.hist(german_bool)
