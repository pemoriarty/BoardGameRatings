#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 08:41:45 2018

@author: pamela
@purpose: parse xmls
"""
from bs4 import BeautifulSoup as bs
#import xmltodict
import pickle
#import lxml

game_ids = range(5000)
#cat_to_pull = "Party Game"#"Children's Game"#"Party Game"
#cat_ids = []
#categories = []
#game_boxes = []
compiled_info = []
#game_names = []

#read in xml
#check if it's a party game
#if so, parse it
#save parsed version

#loop_range = 254656-137000

for game in game_ids:#game_ids:
    file_name = "BoardGameXMLs/xml_info" + str(game_ids[game]) + ".txt"
    fileObject = open(file_name, 'rb')
    xml_content_tmp = fileObject.read()

    soup_tmp = bs(xml_content_tmp, features = 'html5lib')#parse file
    fileObject.close()
    if soup_tmp.find_all('description'):
        year = soup_tmp.find_all('yearpublished')[0].get_text()

        if year != '0':
            #print(str(id) + ' has a year ' + str(year))
            try:
                name = soup_tmp.find_all('name')[0].get_text()
            except IndexError:
                name = None
            
            try:
                minplayers = float(soup_tmp.find_all('minplayers')[0].get_text())  
            except ValueError:
                minplayers = None
            
            try:
                maxplayers = float(soup_tmp.find_all('maxplayers')[0].get_text())   
            except ValueError:
                maxplayers = None
                
            try:
                playtime = float(soup_tmp.find_all('playingtime')[0].get_text())    
            except ValueError:
                playtime = None
            
            try:
                minplaytime = float(soup_tmp.find_all('minplaytime')[0].get_text())   
            except ValueError:
                minplaytime = None
                
            try:
                maxplaytime = float(soup_tmp.find_all('maxplaytime')[0].get_text())    
            except ValueError:
                maxplaytime = None
                
            try:
                age = float(soup_tmp.find_all('age')[0].get_text())
            except ValueError:
                age = None

            description = soup_tmp.find_all('description')[0].get_text()
            
            try:
                image = soup_tmp.find_all('img')[0].get_text()
            except IndexError:
                image = None
            
            try:
                users_rated = float(soup_tmp.find_all('usersrated')[0].get_text())
            except ValueError:
                users_rated = None
            
            try:
                average_rating = float(soup_tmp.find_all('average')[0].get_text())
            except ValueError:
                average_rating = None
                
            try:
                bayes_rating = float(soup_tmp.find_all('bayesaverage')[0].get_text())
            except ValueError:
                bayes_rating = None
                
            try:
                sd_rating = float(soup_tmp.find_all('stddev')[0].get_text())
            except ValueError:
                sd_rating = None
                
            try:
                complexity = float(soup_tmp.find_all('averageweight')[0].get_text())
            except ValueError:
                complexity = None
                
            try:
                num_comp = float(soup_tmp.find_all('numweights')[0].get_text())
            except ValueError:
                num_comp = None
            
            #category = soup_tmp.find_all('boardgamecategory')[0].get_text()
            #mechanic = soup_tmp.find_all('boardgamemechanic')
            #designer = soup_tmp.find_all('boardgamedesigner')
            #version = soup_tmp.find_all('boardgameversion')
            #artist = soup_tmp.find_all('boardgameartist')
            #publisher = soup_tmp.find_all('boardgamepublisher')
        
        
        
            ncat = len(soup_tmp.find_all('boardgamecategory'))#check if the game is categorized      
            if ncat > 0:
                cat_list = []
                for idx in range(ncat):#check if it's a party game
                    cat_list.append(soup_tmp.find_all('boardgamecategory')[idx].get_text())
            
            nmech = len(soup_tmp.find_all('boardgamemechanic'))#check if the game is categorized      
            if nmech > 0:
                mech_list = []
                for idx in range(nmech):#check if it's a party game
                    mech_list.append(soup_tmp.find_all('boardgamemechanic')[idx].get_text())
                    
            npublisher = len(soup_tmp.find_all('boardgamepublisher'))#check if the game is categorized           
            if npublisher > 0:
                publisher_list = []
                for idx in range(npublisher):#check if it's a party game
                    publisher_list.append(soup_tmp.find_all('boardgamepublisher')[idx].get_text())
                
            nsubdomains = len(soup_tmp.find_all('boardgamesubdomain'))
            if nsubdomains > 0:
                subdomain_list = []
                for idx in range(nsubdomains):#check if it's a party game
                    subdomain_list.append(soup_tmp.find_all('boardgamesubdomain')[idx].get_text())
                    
            game_tmp = {'id':game,
                    'name': name,
                    'year': year,
                    'minplayers': minplayers,
                    'maxplayers': maxplayers,
                    'mintime': minplaytime,
                    'maxtime': maxplaytime, 
                    'playtime': playtime,
                    'age': age,
                    'description': description,
                    'users_rated': users_rated,
                    'average_rating': average_rating,
                    'bayes_rating': bayes_rating,
                    'sd_rating': sd_rating,
                    'complexity': complexity,
                    'num_comp': num_comp,
                    'categories': cat_list,
                    'subdomains': subdomain_list,
                    'mechanics': mech_list,
                    'publisher': publisher_list
                    }
                    
            print('game ' + str(game) + ' has been processed')
                #[id, rating_tmp, users_tmp, year_tmp, num_owned_tmp, cat_tmp, img_tmp]
            
            compiled_info.append(game_tmp)

#file_name = "/media/pamela/Stuff/BoardGameXMLs/compiled_info"# + str(game_ids[id]) + ".txt"

file_name = "/media/pamela/Stuff/xmls_parsed2.pickle"
file_name = "/home/pamela/Documents/xmls_parsed"
fileObject = open(file_name,'wb') 
pickle.dump(compiled_info,fileObject)   
fileObject.close()    




