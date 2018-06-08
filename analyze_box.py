#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:02:52 2018

@author: pamela
"""
import cv2
import matplotlib.pyplot as plt

game_id = 499
file_name = "/media/pamela/Stuff/game_images/" + str(game_id) + ".jpg"
img_gray = cv2.imread(file_name, 0)#, cv2.COLOR_BGR2RGB)
img_color = cv2.imread(file_name)
RGB_img = cv2.cvtColor(img_color,cv2.COLOR_BGR2RGB)
#gray_img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)

plt.imshow(RGB_img)#, cmap = 'gray', interpolation = 'bicubic')
plt.imshow(img_gray)

#basic attributes
img_dim = img_tmp.shape
img_size = img_tmp.size

#color attributees
#saturation
HSV_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
plt.imshow(HSV_img)
HSV_img.shape
HSV_img[:,:,2]
img_color[,,2]
HSV_img[:,:,2].mean()

#detect features
fast = cv2.FastFeatureDetector_create()
kp = fast.detect(img_gray, None)
img2 = cv2.drawKeypoints(img_gray, kp, None, color=(255,0,0))
plt.imshow(img2)

#
