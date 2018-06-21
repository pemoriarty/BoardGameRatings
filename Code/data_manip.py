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
file_name = "/home/pamela/Documents/Data/xmls_parsed"
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

df_info2.columns

#make variables for regression: nmech, is_party
nmech = [len(np.asarray(df_info2['mechanics'])[game]) for game in range(df_info2.shape[0])]

sub_df = pd.DataFrame()
sub_df['id'] = df_info2['id']
sub_df['name'] = df_info2['name']
sub_df['ages'] = df_info2['age']
sub_df['nmech'] = nmech
#sub_df['is_party'] = party_bool
sub_df['complexity'] = df_info2['complexity']

##make complexity categorical
np.percentile(sub_df['complexity'], 33)
np.percentile(sub_df['complexity'], 66)

#low: 1-1.8
#mid: 1.81-2.6
#high: 2.6-5

#based on bgg categories:
#low: 1-2
#mid: 2-3.5
#high: 3.5-5
sub_df.reset_index(drop = 0, inplace = True)
sub_df['categorical'] = sub_df['complexity']
mid_index = sub_df['complexity'][(sub_df['complexity'] >=2)]# and (sub_df['complexity'] <=3.5)]
mid_index = mid_index[mid_index <= 3.5].index
low_index = sub_df['complexity'][sub_df['complexity'] < 2].index
high_index = sub_df['complexity'][sub_df['complexity'] > 3.5].index
sub_df['categorical'].replace(sub_df['complexity'][low_index], 'low', inplace = True)
sub_df['categorical'].replace(sub_df['complexity'][mid_index], 'medium', inplace = True)
sub_df['categorical'].replace(sub_df['complexity'][high_index], 'high', inplace = True)


sub_df2 = sub_df.copy(deep = True)

###make matrix of subdomains for each game

subdomains = pd.DataFrame(np.zeros([df_info2.shape[0], 7]), 
                          columns = ['Abstract', 'Thematic', 'Strategy',
                                     'Customizable', 'Party', 'War', 'Family'])
for game in range(df_info2.shape[0]):
    subdomain_tmp = np.asarray(df_info2['subdomains'])[game]
    if 'Abstract Games' in subdomain_tmp:
        subdomains.iloc[game][0] = 1
    if 'Thematic Games' in subdomain_tmp:
        subdomains.iloc[game][1] = 1
    if 'Strategy Games' in subdomain_tmp:
        subdomains.iloc[game][2] = 1
    if 'Customizable Games' in subdomain_tmp:
        subdomains.iloc[game][3] = 1
    if ('Party Games' in subdomain_tmp) or ("Children's Games" in subdomain_tmp):
        subdomains.iloc[game][4] = 1
    #if "Children's Games" in subdomain_tmp:
    #    subdomains.iloc[game][5] = 1
    if 'Wargames' in subdomain_tmp:
        subdomains.iloc[game][5] = 1
    if 'Family Games' in subdomain_tmp:
        subdomains.iloc[game][6] = 1
    
sub_df2 = sub_df2.join(subdomains)


#####make matrix of mechanics for each game#######


file_name = '/home/pamela/Documents/Data/building_model_data'
fileObject = open(file_name, 'wb')
pickle.dump(sub_df2, fileObject)
fileObject.close()

plt.figure()
plt.hist(continuous_y)#approximatley normal!
plt.hist(sub_df2['complexity'])

sub_df2['complexity'].plot('hist')
sub_df2['response'].plot('hist')
##################################
#look at is_party
################### ###############
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

###box saturation
plt.figure()
plt.scatter(sub_df['box_sat'], sub_df['response'])
