#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 20:06:10 2018

@author: pamela
"""

from flask import render_template
from flask import request
from app import app
from PredictComplexity_RF import PredictComplexity2
import pandas as pd
import pickle

@app.route('/')
@app.route('/index')
def game_input():
   
    return render_template("input.html")

@app.route('/weight')
def weight():
  name = request.args.get('game_name')
  return str(PredictComplexity2(name))

@app.route('/output')
def game_output():
  name = request.args.get('game_name')#, 'nmech', 'is_party')

  #check if game is in the database
  the_result = PredictComplexity2(name)
  #return name
  return render_template("output.html", the_result = the_result)
