#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:02:52 2018

@author: pamela
"""
import cv2
import matplotlib.pyplot as plt

game_ids = cat_ids[1:5]

box_attr = []
for id in game_ids:
    d = {}
    d["id"] = id
    file_name = "/media/pamela/Stuff/game_images/" + str(id) + ".jpg"
    img_gray = cv2.imread(file_name, 0)#, cv2.COLOR_BGR2RGB)
    img_color = cv2.imread(file_name)
    
    RGB_img = cv2.cvtColor(img_color,cv2.COLOR_BGR2RGB)

    #plt.imshow(RGB_img)#, cmap = 'gray', interpolation = 'bicubic')
    #plt.imshow(img_gray)

#basic attributes
    d['dim'] = img_color.shape
    d['size'] = img_color.size

#color attributees
#saturation
    HSV_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    #plt.imshow(HSV_img)
    #HSV_img.shape
    #HSV_img[:,:,2]
    #img_color[:,:,2]
    d['S'] = HSV_img[:,:,2].mean()
    
    #b, g, r = cv2.split(img_color)
    b = RGB_img[:,:,1]
    g = RGB_img[:,1,:]
    r = RGB_img[1,:,:]
    d['b'] = b.mean()
    d['g'] = g.mean()
    d['r'] = r.mean()

#detect features
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img_gray, None)
    d['kp'] = img2 = cv2.drawKeypoints(img_gray, kp, None, color=(255,0,0))
    #plt.imshow(img2)
    
    #find countours
    #cv2.findContours(img_gray)

#rect = text_detect(img_gray)
#for i in rect:
#    cv2.rectangle(img_gray,i[:2],i[2:],(0,255,0))


#3cv2.imwrite('img-out.png', img_gray)
    
    box_attr.append(d)
 
#
