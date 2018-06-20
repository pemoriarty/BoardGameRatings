#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:36:27 2018

@author: pamela
"""#        for i in range(len(possible_match)):
#            if best_match[0] == possible_match.iloc[i]:
#                best_idx = possible_idx[i]
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, RandomizedSearchCV, GridSearchCV
from sklearn import ensemble
from sklearn.datasets import make_classification
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
from statsmodels.formula.api import logit, probit, poisson, ols

plt.figure()
pd.plotting.scatter_matrix(sub_df2.drop(columns = {'index', 'id', 'Customizable', 'Abstract', 'Thematic', 'War',  'Family'}))

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
#pd.concat([Xvars, sub_df2.iloc[:, 9:17]], orient = 'columns')

#Xvars.iloc[:, 2:11] = sub_df2.iloc[:, 9:17]
#Xvars['party'] = sub_df2['is_party']
#Xvars['strategy'] = sub_df2['is_strategy']
#Xvars['box_sat'] = sub_df2['box_sat']
#Xvars['child'] = sub_df2['is_child']
#Xvars['nplayers'] = pd.to_numeric(sub_df2['nplayers'])
#Xvars = np.asarray(Xvars)
#Xvars['response'] = sub_df2['response']
yvars = sub_df2['categorical']
#yvars.reset_index(inplace = True)

#Xvars = pd.concat([age_dummies, party_dummies, all_vars['nmech']], axis = 1)
#X_train, X_test, y_train, y_test = train_test_split(Xvars,
#                                                    sub_df2['response'], 
 #                                                   test_size = 0.2)
#print(X_train.shape)
#lm = linear_model.LinearRegression()
#model = lm.fit(X_train, y_train)
#model.coef_
#model.intercept_
#model.score(X_test, y_test)

#split data
np.random.seed(100)
###############sklearn####################
index_nan = Xvars['ages'][np.isnan(Xvars['ages'])].index
Xvars.drop(index_nan, axis = 0, inplace = True)
yvars.drop(index_nan, axis = 0, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(Xvars, yvars, test_size=0.2)
#lm = linear_model.LinearRegression()
lm = linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', max_iter = 200)
model = lm.fit(X_train, y_train)
#model = lm.fit(Xvars, yvars)
model.intercept_
model.coef_
model.score(X_train, y_train)
model.get_params()
model.score(X_test, y_test)

predictions_train = lm.predict(X_train)

len(predictions_train[predictions_train == 'low'])/len(y_train[y_train == 'low'])
len(predictions_train[predictions_train == 'mid'])/len(y_train[y_train == 'mid'])
len(predictions_train[predictions_train == 'high'])/len(y_train[y_train == 'high'])


predictions = lm.predict(X_test)
len(predictions[predictions == 'low'])/len(y_test[y_test == 'low'])
len(predictions[predictions == 'mid'])/len(y_test[y_test == 'mid'])
len(predictions[predictions == 'high'])/len(y_test[y_test == 'high'])

###cross fold validation

scores = cross_val_score(model, Xvars, yvars, cv=10)
print('Cross-validated scores:', scores)

predicted = cross_val_predict(model, Xvars, yvars, cv=5)
chmetrics.accuracy_score(Xvars, predicted) 


plt.figure()
plt.scatter(yvars, predicted)
errors = abs(predictions - y_test)

######################random forest###################3
rf = ensemble.RandomForestClassifier(n_estimators = 100, 
                                     random_state = 42, 
                                     n_jobs = -1, 
                                     min_samples_leaf = 1, 
                                     criterion = 'entropy',
                                     class_weight = 'balanced')
rf = best_grid
# Train the model on training data
rf.get_params()
rf_mod = rf.fit(Xvars, yvars)
scores = cross_val_score(rf_mod, Xvars, yvars, cv = 10)
predicted_cross = cross_val_predict(rf_mod, Xvars, yvars, cv = 10)
metrics.accuracy_score(yvars, predicted_cross)
len(predicted_cross[predicted_cross == 'low'])/len(yvars[yvars == 'low'])
len(predicted_cross[predicted_cross == 'medium'])/len(yvars[yvars == 'medium'])
len(predicted_cross[predicted_cross == 'high'])/len(yvars[yvars == 'high'])

rf.feature_importances_
predictions = rf_mod.predict(Xvars)

rf.oob_score_
sum(predictions == y_test)/len(y_test)
rf_mod.score(X_test, y_test)
rf_mod.score(X_train, y_train)
len(predictions[predictions == 'low'])/len(y_test[y_test == 'low'])
len(predictions[predictions == 'medium'])/len(y_test[y_test == 'medium'])
len(predictions[predictions == 'high'])/len(y_test[y_test == 'high'])#doing a bit better with high complexity games

#####parameter tuning#####
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(3, 110, num = 11)]
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
rf_random = RandomizedSearchCV(estimator = rf_mod, 
                               param_distributions = random_grid, 
                               n_iter = 100, 
                               cv = 3, 
                               verbose=2, 
                               random_state=42, 
                               n_jobs = -1)
# Fit the random search model
rf_random.fit(Xvars, yvars)
rf_random.best_params_

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [90, 100, 110, 120],
    'max_features': ['auto'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [90, 95, 100],
    'criterion': ['entropy']
}
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf_mod, 
                           param_grid = param_grid, 
                          cv = 3, 
                          n_jobs = -1, 
                          verbose = 2)


grid_search.fit(Xvars, yvars)
grid_search.best_params_
{'bootstrap': True,
 'max_depth': 90,
 'max_features': 'auto',
 'min_samples_leaf': 1,
 'min_samples_split': 12,
 'n_estimators': 90}
best_grid = grid_search.best_estimator_


metrics.classification_report(predictions, y_test)
import seaborn as sns
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, predictions)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

export_graphviz(tree,
                feature_names=Xvars.columns,
                filled=True,
                rounded=True)
os.system('dot -Tpng tree.dot -o tree.png')

i_tree = 0
for tree_in_forest in rf.estimators_:
    with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
        my_file = export_graphviz(tree_in_forest, out_file = my_file)
    i_tree = i_tree + 1

file_name = '/home/pamela/Documents/rf_fit_cached'
fileObject = open(file_name, 'wb')
pickle.dump(rf_mod, fileObject)
fileObject.close()
#####################statsmodel below#################
#num_test = int(np.floor(0.2 * Xvars.shape[0]))
#test_idx = np.sort(np.random.choice(range(Xvars.shape[0]), size = num_test, replace = False))

#test = Xvars.iloc[test_idx]

#train = Xvars.copy(deep = True)
#train.reset_index(inplace = True)
#train.drop(test_idx, axis = 0, inplace = True)

#
#
#index_nan = train['ages'][np.isnan(train['ages'])].index
#train.drop(index_nan, axis = 0, inplace = True)

full_vars = X_train.join(y_train)
form = "response ~ ages"
form = "categorical ~ ages + + nmech + C(party)"# + C(strategy)"# + C(child)"
form = "response ~ ages + nmech + C(custom) + C(family) + C(party) + C(strategy) + C(war)"
model_r = smf.ols(formula = form, data = full_vars, missing = 'drop').fit()
print(model_r.summary())

sm.MNLogit(formula = form, data = full_vars).fit()

test = sm.MNLogit(yvars, sm.add_constant(Xvars))
model_test = test.fit()
print(model_test.summary())

res = model_r.resid_pearson
fits = model_r.fittedvalues

fig, ax = plt.subplots()
ax.scatter(fits, res)
ax.hlines(0, -5, 0)

fig, ax = plt.subplots()
ax.scatter(train['ages'], res)
ax.hlines(0, 0, 18)

fig, ax = plt.subplots()
ax.scatter(train['nmech'], res)
ax.hlines(0, 0, 10)

fig, ax = plt.subplots()
ax.scatter(train['party'], res)
ax.hlines(0, 0, 18)

fig, ax = plt.subplots()
ax.scatter(train['strategy'], res)
ax.hlines(0, 0, 18)

fig, ax = plt.subplots(figsize=(8,6))
fig = sm.graphics.influence_plot(model_r, ax=ax)

fig, ax = plt.subplots(figsize=(8,6))
fig = sm.graphics.plot_leverage_resid2(model_r, ax=ax)

fig, ax = plt.subplots(figsize=(8,6))
fig = sm.graphics.plot_fit(model_r, 3 , ax=ax)

fig = sm.qqplot(res, stats.t, fit=True, line='45')


##predict test data
predictions = model_r.predict(test)
#test2 = pd.DataFrame(columns = ['ages', 'nmech', 'party'])
new_data = pd.DataFrame([13, 4, 1], columns = ['ages', 'nmech', 'party'])
new_data = {'ages': 13.0, 'nmech': 4, 'party': 'yes', 'strategy': 'no'}
test2 = pd.DataFrame(new_data, index = [0])


scipy.special.expit(model_r.get_prediction(test2).summary_frame())*4 + 1
scipy.special.expit(model.get_prediction(df_predict).summary_frame())*4 + 1
plt.figure()
plt.scatter(test['response'], predictions)

file_name = '/home/pamela/Documents/model_fit_cached'
fileObject = open(file_name, 'wb')
pickle.dump(model_r, fileObject)
fileObject.close()
