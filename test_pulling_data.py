#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:13:07 2018

@author: pamela
"""

import xmltodict
import urllib


file = urllib.request.urlopen('https://boardgamegeek.com/xmlapi/boardgame/2536?&stats=1')
data = file.read()
file.close()

data_parsed = xmltodict.parse(data)
    
data_parsed['boardgames']['boardgame']['statistics']['ratings']['average']
data_parsed['boardgames']['boardgame']['boardgamemechanic']['#text']


#tried ElementTree library, like xmltodict better- thinking it will be more intuuitive to work with           
#import xml.etree.ElementTree as ET
#
#root = ET.fromstring(data)
#root.tag
#root.attrib
#root = data.getroot()
#
#for child in root:
#    print(child.tag, child.attrib)
#
#root[0][1]
#    
#   doc['mydocument']['@has'] # == u'an attribute'
#doc['mydocument']['and']['many'] # == [u'elements', u'more elements']
#doc['mydocument']['plus']['@a'] # == u'complex'
#doc['mydocument']['plus']['#text'] # == u'element as well'
#    return render_to_response('my_template.html', {'data': data})

###generate random game id's
import random
import numpy
game_ids = random.sample(range(1, 10000), 10)           


ratings = numpy.empty(len(game_ids))
categories = []
images = []
for id in range(len(game_ids)):
    game_xml_string = 'https://boardgamegeek.com/xmlapi/boardgame/' + str(game_ids[id]) + '?&stats=1'

    file2 = urllib.request.urlopen(game_xml_string)
    data2 = file2.read()
    file2.close()

    data_parsed2 = xmltodict.parse(data2)
    
    if 'error' in data_parsed2['boardgames']['boardgame'].keys():
        ratings[id] = None
        categories.append(None)
        images.append(None)
    else:
        ratings[id] = (data_parsed2['boardgames']['boardgame']['statistics']['ratings']['average'])
        #(len(data_parsed2['boardgames']['boardgame']['boardgamemechanic']))
        categories.append(data_parsed2['boardgames']['boardgame']['boardgamemechanic'][0]['#text'])
        images.append(data_parsed2['boardgames']['boardgame']['image'])

from PIL import Image
