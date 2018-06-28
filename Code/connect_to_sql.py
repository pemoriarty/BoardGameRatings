#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:39:05 2018

@author: pamela
@purpose: push database of predicted complexity of all games to sql
"""

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

#set connection info
username = 'postgres'
password = None
host = 'localhost'
port = '5432'
db_name = 'board_games'

engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username,
                       password,
                       host,
                       port,
                       db_name))
print(engine.url)

if not database_exists(engine.url):
    create_database(engine.url)
print(database_exists(engine.url))

#actually push data to SQL
df_info2.to_sql('full_data', engine, if_exists='replace')
