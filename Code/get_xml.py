#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:27:43 2018

@author: pamela
@purpose: read and save boardgamegeek xml files for every game
"""
from api_request import request

game_ids = range(254656)#254656
cat_to_pull = "Party Game"#"Children's Game"#"Party Game"
ids_exist = []
categories = []
images = []
game_info = []
names = []
for ids in game_ids:
    game_xml_string = '''https://boardgamegeek.com/xmlapi/boardgame/'
                        + str(game_ids[ids]) + '?&stats=1'''
    game_tmp = request(game_xml_string, slp=0.5)
    file_name = "/media/pamela/Stuff/BoardGameXMLs/xml_info" + str(game_ids[ids]) + ".txt"
    fileObject = open(file_name, 'wb')
    fileObject.write(game_tmp.content)
    fileObject.close()
