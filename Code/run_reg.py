#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:36:27 2018

@author: pamela
"""
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
#continuous_y ~ ages_dummies + party_dummies + nmech

Xvars = pd.DataFrame()
Xvars['ages'] = ages_tmp
Xvars['nmech'] = all_vars['nmech']
Xvars['party'] = party_bool
Xvars['response'] = continuous_y

Xvars = pd.concat([age_dummies, party_dummies, all_vars['nmech']], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(Xvars,
                                                    continuous_y, 
                                                    test_size = 0.2)
print(X_train.shape)

form = "response ~ nmech"
model_r = smf.OLS(formula = form, data = Xvars).fit()

model = sm.OLS(continuous_y, sm.add_constant(Xvars)).fit()
print(model.summary())
