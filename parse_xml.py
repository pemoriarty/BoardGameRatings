#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 08:41:45 2018

@author: pamela
@purpose: parse xmls
"""
from bs4 import BeautifulSoup as bs
import xmltodict
import random
import numpy
import pickle
import matplotlib.pyplot as plt


game_ids = range(3)
cat_to_pull = "Party Game"#"Children's Game"#"Party Game"
cat_ids = []
categories = []
game_boxes = []
compiled_info = []
game_names = []

#read in xml
#check if it's a party game
#if so, parse it
#save parsed version

for id in game_ids:
    file_name = "BoardGameXMLs/xml_info" + str(game_ids[id]) + ".txt"
    fileObject = open(file_name,'rb')  
    xml_content_tmp = fileObject.read ()

    soup_tmp = bs(xml_content_tmp, 'html5lib')#parse file
    ncat = len(soup_tmp.find_all('boardgamecategory'))#check if the game is categorized
           
    if ncat > 0:
        cat_tmp = []
        for idx in range(ncat):#check if it's a party game
            cat_tmp.append(soup_tmp.find_all('boardgamecategory')[idx].get_text())
        if cat_to_pull in cat_tmp:    
            cat_ids.append(id)
            game_names.append(soup_tmp.find_all('name')[0].get_text())
            xml_tmp = xmltodict.parse(xml_content_tmp.content)

        
            rating_tmp = (xml_tmp['boardgames']['boardgame']['statistics']['ratings']['average'])           
            #if len(soup_tmp.find_all('image')) > 0:
            img_tmp = xml_tmp['boardgames']['boardgame']['image']
            game_tmp = [rating_tmp, cat_tmp, img_tmp]
        
            compiled_info.append(game_tmp)

