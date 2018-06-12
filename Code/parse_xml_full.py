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
import lxml

game_ids = range(254656)
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

for id in range(1000):#game_ids:
    file_name = "BoardGameXMLs/xml_info" + str(game_ids[id]) + ".txt"
    fileObject = open(file_name,'rb')  
    xml_content_tmp = fileObject.read ()

    soup_tmp = bs(xml_content_tmp, features = 'html5lib')#parse file
    fileObject.close()
    if soup_tmp.find_all('description'):
        year = soup_tmp.find_all('yearpublished')[0].get_text()
        if year != '0':
            name = soup_tmp.find_all('name')[0].get_text()
            minplayers = soup_tmp.find_all('minplayers')[0].get_text()    
            maxplayers = soup_tmp.find_all('maxplayers')[0].get_text()   
            playtime = soup_tmp.find_all('playingtime')[0].get_text()    
            minplaytime = soup_tmp.find_all('minplaytime')[0].get_text()    
            maxplaytime = soup_tmp.find_all('maxplaytime')[0].get_text()    
            age = soup_tmp.find_all('age')[0].get_text()
            description = soup_tmp.find_all('description')[0].get_text()
            image = soup_tmp.find_all('img')[0].get_text()
            users_rated = soup_tmp.find_all('usersrated')[0].get_text()
            average_rating = soup_tmp.find_all('average')[0].get_text()
            bayes_rating = soup_tmp.find_all('bayesaverage')[0].get_text()
            sd_rating = soup_tmp.find_all('stddev')[0].get_text()
            complexity = soup_tmp.find_all('averageweight')[0].get_text()
            
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
            
        #        
            #if cat_to_pull in cat_tmp:    
        #            cat_ids.append(id)
        #            game_names.append(soup_tmp.find_all('name')[0].get_text())
        #            xml_tmp = xmltodict.parse(xml_content_tmp)
        #        
        #            users_tmp = (xml_tmp['boardgames']['boardgame']['statistics']['ratings']['usersrated'])
        #            rating_tmp = (xml_tmp['boardgames']['boardgame']['statistics']['ratings']['average'])
        #            year_tmp = (xml_tmp['boardgames']['boardgame']['yearpublished'])
        #            num_owned_tmp = (xml_tmp['boardgames']['boardgame']['statistics']['ratings']['owned'])
        #            #if len(soup_tmp.find_all('image')) > 0:
        #            try:
        #                img_tmp = xml_tmp['boardgames']['boardgame']['image']
        #            except KeyError:
        #                img_tmp = None
                    
                    game_tmp = {'id': id,
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
                                'categories': cat_list,
                                'subdomains': subdomain_list,
                                'mechanics': mech_list,
                                'publisher': publisher_list
                                }
                    
    
                #[id, rating_tmp, users_tmp, year_tmp, num_owned_tmp, cat_tmp, img_tmp]
            
                compiled_info.append(game_tmp)

#file_name = "/media/pamela/Stuff/BoardGameXMLs/compiled_info"# + str(game_ids[id]) + ".txt"

file_name = "Data/compiled_info"
fileObject = open(file_name,'wb') 
pickle.dump(compiled_info,fileObject)   
fileObject.close()    




