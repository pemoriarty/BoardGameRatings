#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:31:32 2018

@author: pamela
"""
import urllib

image_dir = "/media/pamela/Stuff/game_images/"
ngames = len(game_info)
for n in range(ngames):
    name_tmp = ids_exist[n]
    file_name_tmp = image_dir + str(name_tmp) + ".jpg"
    urllib.request.urlretrieve(game_info[n][2], file_name_tmp)

