#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:21:38 2018

@author: pamela
"""
import pandas as pd
import pickle
import difflib
import numpy as np

def PredictComplexity2(game_name):

    file_name = "/home/pamela/Documents/game_complexity_db"
    fileObject = open(file_name, 'rb')
    game_complexity = pickle.load(fileObject)
    fileObject.close()

    best_match = difflib.get_close_matches(game_name, game_complexity['name'].astype(str), n = 1)
    if len(best_match) > 0:
        best_idx = game_complexity['name'][game_complexity['name'].str.contains(best_match[0], na = False)].index
        prediction = np.asarray(game_complexity['complexity'])[best_idx[0]] 
    elif len(best_match) == 0:
       prediction = 'unknown' 

    return prediction