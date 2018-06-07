#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:02:50 2018

@author: pamela
"""
from bs4 import BeautifulSoup as bs
import requests
import time




r = requests.get(game_xml_string)
c = r.content
soup = bs(c, 'html5lib')  # Use the xml parser for API responses and the html_parser for scraping
print(r.status_code)  # 404 not found and the like. Hopefully 200!

soup.find_all('boardgamemechanic')

def request(msg, slp=1):
    '''A wrapper to make robust https requests.'''
    status_code = 500  # Want to get a status-code of 200
    while status_code != 200:
        time.sleep(slp)  # Don't ping the server too often
        try:
            r = requests.get(msg)
            status_code = r.status_code
            if status_code != 200:
                print("Server Error! Response Code %i. Retrying..." % (r.status_code))
        except:
            print("An exception has occurred, probably a momentory loss of connection. Waiting one seconds...")
            time.sleep(1)
    return r


#r2 = request(game_xml_string, slp = 1)
#soup2 = bs(r2.content, 'html5lib')
#soup2.find_all('boardgamemechanic')
game_ids = range(1000)
cat_to_pull = "Party Game"#"Children's Game"#"Party Game"
ids_exist = []
categories = []
images = []
game_info = []
name = []
for id in game_ids:
    game_xml_string = 'https://boardgamegeek.com/xmlapi/boardgame/' + str(game_ids[id]) + '?&stats=1'

    game_tmp = request(game_xml_string, slp = 1)
    soup_tmp = bs(game_tmp.content, 'html5lib')
    ncat = len(soup_tmp.find_all('boardgamecategory'))#check if the game is categorized
           
    if ncat > 0:
        cat_tmp = []
        for idx in range(ncat):#check if it's a children's game
            cat_tmp.append(soup_tmp.find_all('boardgamecategory')[idx].get_text())
        if cat_to_pull in cat_tmp:    
            name.append(soup_tmp.find_all('name')[0].get_text())
            game_xml = xmltodict.parse(game_tmp.content)
            ids_exist.append(id)
        
            rating_tmp = (game_xml['boardgames']['boardgame']['statistics']['ratings']['average'])
            
            #if len(soup_tmp.find_all('image')) > 0:
            img_tmp = game_xml['boardgames']['boardgame']['image']
            #else:
                #img_tmp = None
        
       
            #cat_tmp.append(game_xml['boardgames']['boardgame']['boardgamemechanic'][idx]['#text'])
            game_tmp = [rating_tmp, cat_tmp, img_tmp]
        
            game_info.append(game_tmp)
        
        #img_tmp = soup_tmp.find_all('ratings')['average']
        #ncat = len(data_parsed2['boardgames']['boardgame']['boardgamemechanic'])
        #cat_tmp = []
        