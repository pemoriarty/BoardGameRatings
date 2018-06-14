#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:31:32 2018

@author: pamela
@purpose: retrieve images from a url and save them
"""
import urllib

image_dir = "/media/pamela/Stuff/game_images/"

image_dir = "/home/pamela/game_images"
n_games = len(df_info2)
for n in range(df_info2.shape[0]):
    id_tmp = np.asarray(df_info2['id'])[0]
    file_name_tmp = image_dir + str(id_tmp) + ".jpg"
    try:
        urllib.request.urlretrieve(np.asarray(df_info2['img'])[n], file_name_tmp)
    except (TypeError, urllib.error.HTTPError):
        print(compiled_info[n]['image'])
        print(str(n))


