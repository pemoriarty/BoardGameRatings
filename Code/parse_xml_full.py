#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 08:41:45 2018

@author: pamela
@purpose: parse xmls
"""

import pickle
import numpy as np
from bs4 import BeautifulSoup as bs


game_ids = range(254656)
compiled_info = []

for game in game_ids:#game_ids:
    file_name = "BoardGameXMLs/xml_info" + str(game_ids[game]) + ".txt"
    fileObject = open(file_name, 'rb')
    xml_content_tmp = fileObject.read()

    soup_tmp = bs(xml_content_tmp, features='html5lib')#parse file
    fileObject.close()
    if soup_tmp.find_all('description'):
        year = soup_tmp.find_all('yearpublished')[0].get_text()

        if year != '0':
            try:
                name = soup_tmp.select('name[primary]')[0].get_text()
            except IndexError:
                name = np.nan

            try:
                minplayers = float(soup_tmp.find_all('minplayers')[0].get_text())
            except ValueError:
                minplayers = np.nan

            try:
                maxplayers = float(soup_tmp.find_all('maxplayers')[0].get_text())
            except ValueError:
                maxplayers = np.nan

            try:
                playtime = float(soup_tmp.find_all('playingtime')[0].get_text())
            except ValueError:
                playtime = np.nan

            try:
                minplaytime = float(soup_tmp.find_all('minplaytime')[0].get_text())
            except ValueError:
                minplaytime = np.nan

            try:
                maxplaytime = float(soup_tmp.find_all('maxplaytime')[0].get_text())
            except ValueError:
                maxplaytime = np.nan

            try:
                age = float(soup_tmp.find_all('age')[0].get_text())
            except ValueError:
                age = np.nan

            description = soup_tmp.find_all('description')[0].get_text()

            try:
                image = soup_tmp.find_all('thumbnail')[0].get_text()
            except IndexError:
                image = np.nan

            try:
                users_rated = float(soup_tmp.find_all('usersrated')[0].get_text())
            except ValueError:
                users_rated = np.nan

            try:
                average_rating = float(soup_tmp.find_all('average')[0].get_text())
            except ValueError:
                average_rating = np.nan

            try:
                bayes_rating = float(soup_tmp.find_all('bayesaverage')[0].get_text())
            except ValueError:
                bayes_rating = np.nan

            try:
                sd_rating = float(soup_tmp.find_all('stddev')[0].get_text())
            except ValueError:
                sd_rating = np.nan

            try:
                complexity = float(soup_tmp.find_all('averageweight')[0].get_text())
            except ValueError:
                complexity = np.nan

            try:
                num_comp = float(soup_tmp.find_all('numweights')[0].get_text())
            except ValueError:
                num_comp = np.nan

            ncat = len(soup_tmp.find_all('boardgamecategory'))#check if the game is categorized
            if ncat > 0:
                cat_list = []
                for idx in range(ncat):#check if it's a party game
                    cat_list.append(soup_tmp.find_all('boardgamecategory')[idx].get_text())

            nmech = len(soup_tmp.find_all('boardgamemechanic'))
            if nmech > 0:
                mech_list = []
                for idx in range(nmech):#check if it's a party game
                    mech_list.append(soup_tmp.find_all('boardgamemechanic')[idx].get_text())

            npublisher = len(soup_tmp.find_all('boardgamepublisher'))
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
                        'image': image,
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

file_name = "/media/pamela/Stuff/xmls_parsed"
#file_name = "/home/pamela/Documents/xmls_parsed"
fileObject = open(file_name, 'wb')
pickle.dump(compiled_info, fileObject)
fileObject.close()
