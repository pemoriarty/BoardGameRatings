#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:27:43 2018

@author: pamela
@purpose: read and save boardgamegeek xml files for every game
"""

from bs4 import BeautifulSoup as bs
import requests
import time
import xmltodict
import random
import numpy
import pickle
import

from api_request import request

r = requests.get(game_xml_string)

game_ids = range(254656)#254656
cat_to_pull = "Party Game"#"Children's Game"#"Party Game"
ids_exist = []
categories = []
images = []
game_info = []
names = []
for id in game_ids:
    game_xml_string = 'https://boardgamegeek.com/xmlapi/boardgame/' + str(game_ids[id]) + '?&stats=1'
    game_tmp = request(game_xml_string, slp = 1)
    #soup_tmp = bs(game_tmp.content, 'html5lib')
    #game_xml = xmltodict.parse(game_tmp.content)
    file_name = "/media/pamela/Stuff/BoardGameXMLs/xml_info" + str(game_ids[id]) + ".txt"
    fileObject = open(file_name,'wb') 
    #pickle.dump((soup_tmp, game_xml),fileObject)   
    fileObject.write(game_tmp.content)
    fileObject.close()    
