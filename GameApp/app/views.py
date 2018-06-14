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

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Miguel' },
       )

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

@app.route('/input')
def game_input():
    return render_template("input.html")

@app.route('/output')
def game_output():
  #pull 'birth_month' from input field and store it
  age = request.args.get('age')#, 'nmech', 'is_party')
  nmech = request.args.get('nmech')
  is_party = request.args.get('is_party')
  is_strategy = request.args.get('is_strategy')
    #just select the Cesareans  from the birth dtabase for the month that the user inputs
  #query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
  #print(query)
  #query_results=pd.read_sql_query(query,con)
  #print(query_results)
  #births = []
  #for i in range(0,query_results.shape[0]):
  #    births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
  the_result = PredictComplexity(age, nmech, is_party, is_strategy)
  return render_template("output.html", the_result = the_result)
