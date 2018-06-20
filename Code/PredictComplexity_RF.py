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

def PredictComplexity(game_name):
    import pickle
    import difflib
    
#    file_name = '/home/pamela/Documents/game_complexity_db'
#    fileObject = open(file_name, 'rb')
#    games = pickle.load(fileObject)
#    fileObject.close()

    possible_match = df_info2['name'][df_info2['name'].str.contains(game_name, case = False, na = False)]
    if len(possible_match) > 1:
        possible_idx = df_info2['name'][df_info2['name'].str.contains(game_name, case = False, na = False)].index
        best_match = difflib.get_close_matches(game_name, possible_match, n=1)
        if len(best_match) > 0:
            for i in range(len(possible_match)):
                if best_match[0] == possible_match.iloc[i]:
                    best_idx = possible_idx[i]
            prediction = np.asarray(df_info2['complexity'])[best_idx] 
        elif len(best_match) == 0:
           prediction = np.asarray(df_info2['complexity'])[possible_idx[0]] 
    elif len(possible_match == 1):
        possible_idx = df_info2['name'][df_info2['name'].str.contains(game_name, case = False, na = False)].index
        prediction = np.asarray(df_info2['complexity'])[possible_idx[0]] 
    else:
        prediction = 'unknown'


    #prediction_out = pd.DataFrame()
    return prediction