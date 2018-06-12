#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:36:27 2018

@author: pamela
"""
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
#continuous_y ~ ages_tmp
X_train, X_test, y_train, y_test = train_test_split(ages_tmp.reshape(-1, 1), 
                                                    continuous_y, 
                                                    test_size = 0.2)
print(X_train.shape)

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
model.coef_
model.intercept_
model.score(X_test, y_test)
