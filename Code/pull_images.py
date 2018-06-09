#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:31:32 2018

@author: pamela
@purpose: retrieve images from a url and save them
"""
import urllib

image_dir = "/media/pamela/Stuff/game_images/"
n_games = len(compiled_info)
for n in range(3129,n_games):
    id_tmp = compiled_info[n]['id']
    file_name_tmp = image_dir + str(id_tmp) + ".jpg"
    try:
        urllib.request.urlretrieve(compiled_info[n]['image'], file_name_tmp)
    except (TypeError, urllib.error.HTTPError):
        print(compiled_info[n]['image'])
        print(str(n))


