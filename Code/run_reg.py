#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:36:27 2018

@author: pamela
@purpose: run random forest and plot validation
"""



import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn import datasets, metrics
from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     cross_val_predict,
                                     RandomizedSearchCV,
                                     GridSearchCV)
from sklearn import ensemble
from sklearn.feature_selection import RFECV
from statsmodels.formula.api import ols
from imblearn.over_sampling import SMOTE


file_name = '/home/pamela/Documents/Data/building_model_data'
fileObject = open(file_name, 'rb')
sub_df2 = pickle.load(fileObject)
fileObject.close()

plt.figure()
pd.plotting.scatter_matrix(sub_df2.drop(columns={'index',
                                                 'id',
                                                 'Customizable',
                                                 'Abstract',
                                                 'Thematic',
                                                 'War', 
                                                 'Family'}))

Xvars = pd.DataFrame()
Xvars['ages'] = pd.to_numeric(sub_df2['ages'])
Xvars['nmech'] = sub_df2['nmech']
Xvars['abstract'] = sub_df2['Abstract']
Xvars['thematic'] = sub_df2['Thematic']
Xvars['war'] = sub_df2['War']
Xvars['custom'] = sub_df2['Customizable']
Xvars['family'] = sub_df2['Family']
Xvars['party'] = sub_df2['Party']
#Xvars['child'] = sub_df2['Child']
Xvars['strategy'] = sub_df2['Strategy']

yvars = sub_df2['categorical']
#yvars.reset_index(inplace = True)

#split data
np.random.seed(100)
###############sklearn####################
index_nan = Xvars['ages'][np.isnan(Xvars['ages'])].index
Xvars.drop(index_nan, axis=0, inplace=True)
yvars.drop(index_nan, axis=0, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(Xvars, yvars, test_size=0.2)
sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_sample(Xvars, yvars)
#lm = linear_model.LinearRegression()

######################random forest###################
###tune hyperparameters
rf = ensemble.RandomForestClassifier(n_estimators=100,
                                     random_state=42,
                                     n_jobs=-1,
                                     min_samples_leaf=1,
                                     criterion='entropy',
                                     #class_weight = 'balanced')
                                    )
rf_mod = rf.fit(X_sm, y_sm)
n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(3, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
criteria = ['gini', 'entropy']
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criteria}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf_mod,
                               param_distributions=random_grid,
                               n_iter=100,
                               cv=3,
                               verbose=2,
                               random_state=42,
                               n_jobs=-1)
# Fit the random search model
rf_random.fit(X_sm, y_sm)
rf_random.best_params_

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [90, 100, 110, 120],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [90, 95, 100],
    'criterion': ['gini']
}
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf_mod,
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=2)

grid_search.fit(Xvars, yvars)
grid_search.best_params_
{'bootstrap': True,
 'max_depth': 90,
 'max_features': 'sqrt',
 'min_samples_leaf': 1,
 'min_samples_split': 4,
 'n_estimators': 90}
best_grid = grid_search.best_estimator_

#train the model on training data
rf = best_grid
rf = ensemble.RandomForestClassifier(bootstrap=True,
                                     criterion='gini',
                                     max_depth=90,
                                     max_features='sqrt',
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     min_impurity_split=None,
                                     min_samples_leaf=3,
                                     min_samples_split=4,
                                     min_weight_fraction_leaf=0.0,
                                     n_estimators=200,
                                     n_jobs=-1,
                                     oob_score=False,
                                     random_state=42,
                                     verbose=0,
                                     warm_start=False,
                                     class_weight='balanced')
rf.get_params()
rf_mod = rf.fit(X_sm, y_sm)
rf_mod = rf.fit(X_train, y_train)
scores = cross_val_score(rf_mod, X_sm, y_sm, cv=10)
predicted_cross = cross_val_predict(rf_mod, X_sm, y_sm, cv = 10)
metrics.accuracy_score(y_sm, predicted_cross)
len(predicted_cross[predicted_cross == 'low'])/len(y_sm[y_sm == 'low'])
len(predicted_cross[predicted_cross == 'medium'])/len(y_sm[y_sm == 'medium'])
len(predicted_cross[predicted_cross == 'high'])/len(y_sm[y_sm == 'high'])

selector = RFECV(rf_mod)
select2 = selector.fit(X_sm, y_sm)
select2.support_


Xvars.columns
print(rf.feature_importances_)
predictions = rf_mod.predict(X_sm)
predictions_test = rf_mod.predict(X_test)

sum(y_test == predictions_test)/len(y_test)
len(predictions_test[predictions_test == 'low'])/len(y_test[y_test == 'low'])
len(predictions_test[predictions_test == 'medium'])/len(y_test[y_test == 'medium'])
len(predictions_test[predictions_test == 'high'])/len(y_test[y_test == 'high'])

print(metrics.classification_report(predicted_cross, y_sm))
#metrics.roc_curve(onehot_encoded, predictions)

file_name = '/home/pamela/Documents/Data/rf_fit_cached'
fileObject = open(file_name, 'wb')
pickle.dump(rf_mod, fileObject)
fileObject.close()

#############smote

len(y_res[y_res == 'high'])
len(y_res[y_res == 'medium'])
len(y_res[y_res == 'low'])


#plot confusion matrix
file_name = '/home/pamela/Documents/Data/rf_fit_cached'
fileObject = open(file_name, 'rb')
rf_mod = pickle.load(fileObject)
fileObject.close()

mat = confusion_matrix(y_sm, predicted_cross, labels=['low', 'medium', 'high'])

plt.figure()
sns.set(font_scale=3)
sns.palplot(sns.color_palette("Blues"))
sns.palplot(sns.color_palette("GnBu_d"))
sns.palplot(sns.cubehelix_palette(8, start=.5, rot=-.75))
cm = sns.heatmap(mat.T,
                 square=True,
                 annot=False,
                 fmt='d',
                 cbar=True,
                 cmap='YlGnBu')
cm.set_xlabel("True", fontsize=50)
plt.xticks(rotation=0)
cm.set_xticklabels(labels=['low', 'medium', 'high'])
plt.yticks(rotation=90)
cm.set_yticklabels(labels=['low', 'medium', 'high'])
cm.set_ylabel("Predicted",nfontsize=60)
cm.tick_params(labelsize=50)
plt.show()

sns.choose_colorbrewer_palette('sequential')

###make theoretical perfect heatmap
perfect = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
plt.figure()
sns.heatmap(perfect)
cm_perf = sns.heatmap(perfect,
                      square=True,
                      annot=False,
                      fmt='d',
                      cbar=True,
                      cmap='Blues')
cm_perf.set_xlabel("True", fontsize=50)
plt.xticks(rotation=0)
cm_perf.set_xticklabels(labels=['low', 'medium', 'high'])
plt.yticks(rotation=90)
cm_perf.set_yticklabels(labels=['low', 'medium', 'high'])
cm_perf.set_ylabel("Predicted", fontsize=60)
cm_perf.tick_params(labelsize=50)


##base case
test_x = pd.DataFrame()
test_x['ages'] = sub_df['ages'].astype('float')
test_x['playtime'] = sub_df['playtime'].astype('float')

test = sm.OLS(sub_df['complexity'].astype('float'), (test_x), missing='drop').fit()
print(test.summary())
