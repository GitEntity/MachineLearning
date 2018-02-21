# Author: Devante Wilson
# Date: February 18, 2018
#
# This program is an NBA shot classifier for the regular 2015-2016 season.

# import required libraries
from __future__ import print_function, division
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
palette = sns.color_palette('deep', 5)
palette[1], palette[2] = palette[2], palette[1]

# load the NBA shot data set using Pandas
nba = pd.read_csv('train.csv')

# show heatmap of missing values
# plt.figure(figsize=(12, 12))
# sns.heatmap(nba.isnull(), yticklabels=False, cbar=False, cmap='jet')
# plt.show()

# # check missing data (final check)
# print(nba[nba.isnull().any(axis=1)])

# convert categorical classes into binary values to feed to the model
action_type = pd.get_dummies(nba['ACTION_TYPE'])
shot_type = pd.get_dummies(nba['SHOT_TYPE'])
shot_zone_area = pd.get_dummies(nba['SHOT_ZONE_AREA'])
shot_zone_basic = pd.get_dummies(nba['SHOT_ZONE_BASIC'])
shot_zone_range = pd.get_dummies(nba['SHOT_ZONE_RANGE'])

# remove old features
nba = nba.drop(['ACTION_TYPE', 'EVENTTIME', 'EVENT_TYPE', 'GAME_DATE', 'GAME_EVENT_ID',
                'GAME_ID', 'HTM', 'MINUTES_REMAINING', 'PERIOD', 'PLAYER_ID', 'PLAYER_NAME',
                'QUARTER', 'SECONDS_REMAINING', 'SHOT_ATTEMPTED_FLAG', 'SHOT_TIME', 'SHOT_TYPE',
                'SHOT_ZONE_AREA', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_RANGE', 'TEAM_ID', 'TEAM_NAME', 'VTM', 'LOC_X', 'LOC_Y'], axis=1)
# add all of the new features
nba = pd.concat([nba, action_type, shot_type, shot_zone_area, shot_zone_basic, shot_zone_range], axis=1)

# define prediction target
y = nba.SHOT_MADE_FLAG
# set prediction data (return view/copy with column removed)
X = nba.drop('SHOT_MADE_FLAG', axis=1)

# create a randomized train and test split with scikit-learn
# and withhold all test data from model when training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# create logistic regression model
logmodel = LogisticRegression()

# fit/train the model on the training data
logmodel.fit(X_train, y_train)

# generate predictions for the withheld test data
predictions = logmodel.predict(X_test)

# evaluate performance of the model
print(classification_report(y_test, predictions))
print("Accuracy: {}".format(accuracy_score(y_test, predictions)))

# # standard scale all the training data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
#
# # create Support Vector Machine/Classifier (SVM/SVC) model
# svmmodel = SVC()
#
# # fit/train the model on the training data
# svmmodel.fit(X_train_scaled, y_train)
#
# # scale test data using standard scaler
# X_test_scaled = scaler.transform(X_test)
#
# # generate predictions
# predictions = svmmodel.predict(X_test_scaled)
#
# # evaluate performance of the model
# print(classification_report(y_test, predictions))
# print("Accuracy: {}".format(accuracy_score(y_test, predictions)))