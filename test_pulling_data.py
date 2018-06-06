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

           