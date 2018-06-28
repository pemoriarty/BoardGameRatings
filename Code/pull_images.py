#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:31:32 2018

@author: pamela
@purpose: retrieve images from a url and save them
"""
import urllib

image_dir = "/media/pamela/Stuff/game_images/"

n_games = df_info2.shape[0]
for n in range(n_games):
    id_tmp = np.asarray(df_info2['id'])[n]
    file_name_tmp = image_dir + str(id_tmp) + ".jpg"
    try:
        urllib.request.urlretrieve(np.asarray(df_info2['image'])[n], file_name_tmp)
    except (TypeError, ValueError, urllib.error.HTTPError):
        print(np.asarray(df_info2['image'])[n])
        print(str(n))
