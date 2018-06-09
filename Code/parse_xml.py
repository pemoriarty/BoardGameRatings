#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 08:41:45 2018

@author: pamela
@purpose: parse xmls
"""
from bs4 import BeautifulSoup as bs
import xmltodict
import pickle

game_ids = range(75000, 137000)
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

loop_range = 137000-75000

for id in range(loop_range):#game_ids:
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
            xml_tmp = xmltodict.parse(xml_content_tmp)
        
            users_tmp = (xml_tmp['boardgames']['boardgame']['statistics']['ratings']['usersrated'])
            rating_tmp = (xml_tmp['boardgames']['boardgame']['statistics']['ratings']['average'])
            year_tmp = (xml_tmp['boardgames']['boardgame']['yearpublished'])
            num_owned_tmp = (xml_tmp['boardgames']['boardgame']['statistics']['ratings']['owned'])
            #if len(soup_tmp.find_all('image')) > 0:
            try:
                img_tmp = xml_tmp['boardgames']['boardgame']['image']
            except KeyError:
                img_tmp = None
            game_tmp = {'id': id,
                        'rating': rating_tmp,
                        'no_ratings': users_tmp,
                        'year': year_tmp,
                        'no_owned': num_owned_tmp,
                        'categories': cat_tmp,
                        'image': img_tmp}
            #[id, rating_tmp, users_tmp, year_tmp, num_owned_tmp, cat_tmp, img_tmp]
        
            compiled_info.append(game_tmp)

file_name = "/media/pamela/Stuff/BoardGameXMLs/compiled_info"# + str(game_ids[id]) + ".txt"
file_name = "compiled_info"
fileObject = open(file_name,'wb') 
pickle.dump(compiled_info,fileObject)   
fileObject.close()    


