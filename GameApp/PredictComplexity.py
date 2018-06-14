#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:21:38 2018

@author: pamela
"""




def PredictComplexity(age, nmech, is_party, is_strategy):
    import pickle
    import pandas as pd
    import scipy.special
    
    file_name = '/home/pamela/Documents/model_fit_cached'
    fileObject = open(file_name, 'rb')
    model = pickle.load(fileObject)
    fileObject.close()

#    x_to_predict = pd.DataFrame(columns = ['ages', 'nmech', 'party'])
#    x_to_predict['ages'] = age
#    x_to_predict['nmech'] = nmech
#    x_to_predict['party'] = is_party
    
    #new_data = pd.DataFrame([age, nmech, is_party], columns = ['ages', 'nmech', 'party'])
    #new_data = pd.DataFrame([age, nmech, is_party], columns = ['ages', 'nmech', 'party'], index = [0])
    if is_party == 'yes' or is_party == 'Yes':
        party_num = 1
    else:
        party_num = 0
        
    if is_strategy == 'yes' or is_strategy == 'Yes':
        strategy_num = 1
    else:
        strategy_num = 0
    
    new_data = {'ages': float(age), 'nmech':float(nmech), 'party': float(party_num), 'strategy': float(strategy_num)}
    df_predict = pd.DataFrame(new_data, index = [0])
    prediction = model.get_prediction(df_predict)
    est = prediction.summary_frame()['mean']
    SE = prediction.summary_frame()['mean_se']
    lower_ci = prediction.summary_frame()['obs_ci_lower']
    upper_ci = prediction.summary_frame()['obs_ci_upper']

    transformed_est = round((scipy.special.expit(est)*4 + 1)[0], 2)
    transformed_lower_ci = round((scipy.special.expit(lower_ci)*4 + 1)[0], 2)
    transformed_upper_ci = round((scipy.special.expit(upper_ci)*4 + 1)[0], 2)

    #prediction_out = pd.DataFrame()
    return [transformed_est, transformed_lower_ci, transformed_upper_ci]