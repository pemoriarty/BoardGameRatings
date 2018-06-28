#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:29:58 2018

@author: pamela
@purpose: slow down rate of api requests
"""
import time

def request(msg, slp=0.5):
    '''A wrapper to make robust https requests.'''
    status_code = 500  # Want to get a status-code of 200
    while status_code != 200:
        time.sleep(slp)  # Don't ping the server too often
        try:
            r = requests.get(msg)
            status_code = r.status_code
            if status_code != 200:
                print("Server Error! Response Code %i. Retrying..." % (r.status_code))
        except:
            print("""An exception has occurred, probably a momentory loss of connection.
                  Waiting one seconds...""")
            time.sleep(1)
    return r
