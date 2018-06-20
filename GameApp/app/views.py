#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 20:06:10 2018

@author: pamela
"""

from flask import render_template
from flask import request
from app import app
from PredictComplexity import PredictComplexity
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import pickle
#import psycopg2
#import requests

#user = 'postgres' #add your username here (same as previous postgreSQL)                      
#host = 'localhost'
#password = None
#dbname = 'birth_db'
#port = 5430
#db = create_engine('postgres://{}:{}@{}:{}/{}'.format(user, password, host, port, dbname))

#con = None
#con = psycopg2.connect(database = dbname, user = user, host = host, port = port)

#@app.route('/db')
#def birth_page():
#    sql_query = """                                                                       
#                SELECT * FROM birth_data_table WHERE delivery_method='Cesarean';          
#                """
#    query_results = pd.read_sql_query(sql_query,con)
#    births = ""
#    for i in range(0,10):
#        births += query_results.iloc[i]['birth_month']
#        births += "<br>"
#    return births
@app.route('/')
@app.route('/index')
def game_input():
   
    return render_template("input.html")


@app.route('/output')
def game_output():
  name = request.args.get('game_name')#, 'nmech', 'is_party')

  #check if game is in the database
  the_result = PredictComplexity_RF(name)
  return name
  #return render_template("output.html", the_result = the_result)
