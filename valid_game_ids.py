#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:02:50 2018

@author: pamela
"""
from bs4 import BeautifulSoup as bs
import requests
import time
import xmltodict
import random
import numpy
import pickle
import matplotlib.pyplot as plt

from api_request import api_request

game_ids = range(100)
cat_to_pull = "Party Game"#"Children's Game"#"Party Game"
ids_exist = []
categories = []
images = []
game_info = []
names = []
for id in game_ids:
    game_xml_string = 'https://boardgamegeek.com/xmlapi/boardgame/' + str(game_ids[id]) + '?&stats=1'

    game_tmp = api_request(game_xml_string, slp = 1)
    soup_tmp = bs(game_tmp.content, 'html5lib')
    game_xml = xmltodict.parse(game_tmp.content)
    file_name = "/media/pamela/Stuff/BoardGameXMLs/xml_info" + str(game_ids[id])# +".p"
    fileObject = open(file_name,'wb') 
    pickle.dump((soup_tmp, game_xml),fileObject)   
    fileObject.close()    
    
    
    ncat = len(soup_tmp.find_all('boardgamecategory'))#check if the game is categorized
           
    if ncat > 0:
        cat_tmp = []
        for idx in range(ncat):#check if it's a children's game
            cat_tmp.append(soup_tmp.find_all('boardgamecategory')[idx].get_text())
        if cat_to_pull in cat_tmp:    
            names.append(soup_tmp.find_all('name')[0].get_text())
            game_xml = xmltodict.parse(game_tmp.content)
            ids_exist.append(id)
        
            rating_tmp = (game_xml['boardgames']['boardgame']['statistics']['ratings']['average'])
            
            #if len(soup_tmp.find_all('image')) > 0:
            img_tmp = game_xml['boardgames']['boardgame']['image']
 
            game_tmp = [rating_tmp, cat_tmp, img_tmp]
        
            game_info.append(game_tmp)
  
#file_name = "Documents/BoardGameRatings/game_info"        
file_name = "game_info"
fileObject = open(file_name,'wb') 
pickle.dump((game_info, ids_exist, name) ,fileObject)   
fileObject.close()

fileObject = open(file_name,'rb')  
games_test, ids_test, name_test = pickle.load(fileObject) 
soup_tmp, game_xml = pickle.load(fileObject) 
#

