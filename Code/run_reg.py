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
Xvars['ages'] = pd.to_numeric(sub_df2['ages'])
Xvars['nmech'] = sub_df2['nmech']
Xvars['party'] = sub_df2['is_party']
#Xvars = np.asarray(Xvars)
Xvars['response'] = sub_df2['response']
Xvars.reset_index(inplace = True)

#Xvars = pd.concat([age_dummies, party_dummies, all_vars['nmech']], axis = 1)
#X_train, X_test, y_train, y_test = train_test_split(Xvars,
#                                                    sub_df2['response'], 
                                                    test_size = 0.2)
#print(X_train.shape)
#lm = linear_model.LinearRegression()
#model = lm.fit(X_train, y_train)
#model.coef_
#model.intercept_
#model.score(X_test, y_test)

#split data
np.random.seed(100)
num_test = int(np.floor(0.33 * Xvars.shape[0]))
test_idx = np.sort(np.random.choice(range(Xvars.shape[0]), size = num_test, replace = False))

test = Xvars.iloc[test_idx]

train = Xvars.copy(deep = True)
train.reset_index(inplace = True)
train.drop(test_idx, axis = 0, inplace = True)


form = "response ~ ages + nmech + party"
model_r = smf.ols(formula = form, data = train, missing = 'drop').fit()
print(model_r.summary())

