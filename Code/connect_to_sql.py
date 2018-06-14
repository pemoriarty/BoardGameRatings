#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:39:05 2018

@author: pamela
"""

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd

# Define a database name (we're using a dataset on births, so we'll call it birth_db)
# Set your postgres username/password, and connection specifics
username = 'postgres'
password = None    # change this
host     = 'localhost'
port     = '5432'            # default port that postgres listens on
db_name  = 'board_games'

engine = create_engine( 'postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, db_name) )
print(engine.url)

if not database_exists(engine.url):
    create_database(engine.url)
print(database_exists(engine.url))

df_info2.to_sql('full_data', engine, if_exists='replace')
