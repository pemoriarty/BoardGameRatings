#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:21:38 2018

@author: pamela
"""

import pandas as pd
#import scipy.special
def FindGame(game_to_match,game_df):
    game_df['name'][game_to_match in game_df['name']]
    
    return game_id

def GameInfo(game_id):
    
    return [age, nmech, is_family, is_party, is_strategy, is_abstract, is_thematic, is_war]

def PredictComplexity(age, nmech, is_family, is_party, is_strategy, is_abstract, is_thematic, is_war):
    import pickle
    
    file_name = '/home/pamela/Documents/rf_fit_cached'
    fileObject = open(file_name, 'rb')
    pickle.load(fileObject)
    fileObject.close()

#    x_to_predict = pd.DataFrame(columns = ['ages', 'nmech', 'party'])
#    x_to_predict['ages'] = age
#    x_to_predict['nmech'] = nmech
#    x_to_predict['party'] = is_party
    
    #new_data = pd.DataFrame([age, nmech, is_party], columns = ['ages', 'nmech', 'party'])
    #new_data = pd.DataFrame([age, nmech, is_party], columns = ['ages', 'nmech', 'party'], index = [0])
#    if is_party == 'yes' or is_party == 'Yes':
#        party_num =1
#    else:
#        party_num = 0
    new_data = {'ages': age, 'nmech':nmech, 'family': is_family, 
                'party': is_party, 'strategy': is_strategy, 
                'abstract': is_abstract, 'thematic': is_thematic, 'war': is_war}
    df_predict = pd.DataFrame(new_data, index = [0])
    prediction = rf_mod.predict(df_predict)
#    est = prediction.summary_frame()['mean']
#    SE = prediction.summary_frame()['mean_se']
#    lower_ci = prediction.summary_frame()['obs_ci_lower']
#    upper_ci = prediction.summary_frame()['obs_ci_upper']
#
#    transformed_est = (scipy.special.expit(est)*4 + 1)[0]
#    transformed_lower_ci = (scipy.special.expit(lower_ci)*4 + 1)[0]
#    transformed_upper_ci = (scipy.special.expit(upper_ci)*4 + 1)[0]

    #prediction_out = pd.DataFrame()
    return prediction