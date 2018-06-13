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
import scipy.stats as stats
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


form = "response ~ ages + nmech + C(party)"
model_r = smf.ols(formula = form, data = train, missing = 'drop').fit()
print(model_r.summary())

res = model_r.resid_pearson
fits = model_r.fittedvalues

fig, ax = plt.subplots()
ax.scatter(fits, res)
ax.hlines(0, 0, 1)

fig, ax = plt.subplots(figsize=(8,6))
fig = sm.graphics.influence_plot(model_r, ax=ax)

fig, ax = plt.subplots(figsize=(8,6))
fig = sm.graphics.plot_leverage_resid2(model_r, ax=ax)

fig, ax = plt.subplots(figsize=(8,6))
fig = sm.graphics.plot_fit(model_r, 3, ax=ax)

fig = sm.qqplot(res, stats.t, fit=True, line='45')


##predict test data
predictions = model_r.predict(test)
#test2 = pd.DataFrame(columns = ['ages', 'nmech', 'party'])
new_data = pd.DataFrame([13, 4, 1], columns = ['ages', 'nmech', 'party'])
new_data = {'ages': 13.0, 'nmech': 4, 'party': 1}
test2 = pd.DataFrame(new_data, index = [0])
model_r.predict(test2)

plt.figure()
plt.scatter(test['response'], predictions)

file_name = '/home/pamela/Documents/model_fit_cached'
fileObject = open(file_name, 'wb')
pickle.dump(model_r, fileObject)
fileObject.close()
