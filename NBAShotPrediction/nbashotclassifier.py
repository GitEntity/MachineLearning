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

# detect the use of the test set to adjust for missing header (SHOT_MADE_FLAG)
def test_data_loaded(data_frame):
    """The 'SHOT_MADE_FLAG' header is not present in the test set (solution_no_answer.csv)
        and thus, cannot be indexed. When dropping old features, this header must be removed from the list.
        The 'EVENT_TYPE' header is the same thing, so it is used instead."""
    # drop header from the data frame
    data_frame = data_frame.drop(['ACTION_TYPE', 'EVENTTIME', 'GAME_DATE', 'HTM',
                'MINUTES_REMAINING', 'PERIOD', 'PLAYER_ID', 'PLAYER_NAME', 'QUARTER', 'SECONDS_REMAINING',
                'SHOT_ATTEMPTED_FLAG', 'SHOT_TIME', 'SHOT_TYPE', 'SHOT_ZONE_AREA', 'SHOT_ZONE_BASIC',
                'SHOT_ZONE_RANGE', 'TEAM_ID', 'TEAM_NAME', 'VTM', 'LOC_X', 'LOC_Y'], axis=1)
    return data_frame

# define file name
file_name = 'train.csv'
# load the NBA shot data set using Pandas
nba = pd.read_csv(file_name)

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

# detect use of test set
if file_name=='solution_no_answer.csv':
    nba = test_data_loaded(nba)
else: # training or validation set are being used
    # remove old features
    nba = nba.drop(['ACTION_TYPE', 'EVENTTIME', 'GAME_DATE', 'GAME_ID', 'HTM',
                    'MINUTES_REMAINING', 'PERIOD', 'PLAYER_ID', 'PLAYER_NAME', 'QUARTER', 'SECONDS_REMAINING',
                    'SHOT_ATTEMPTED_FLAG', 'SHOT_TIME', 'SHOT_TYPE', 'SHOT_ZONE_AREA', 'SHOT_ZONE_BASIC',
                    'SHOT_ZONE_RANGE', 'TEAM_ID', 'TEAM_NAME', 'VTM', 'LOC_X', 'LOC_Y'], axis=1)

# add all of the new features
nba = pd.concat([nba, action_type, shot_type, shot_zone_area, shot_zone_basic, shot_zone_range], axis=1)

# define prediction target
y = nba['SHOT_MADE_FLAG']
# set prediction data (return view/copy with column(s) removed)
X = nba.drop(['SHOT_MADE_FLAG', 'GAME_EVENT_ID'], axis=1)

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# create logistic regression model (transformed = 1 / (1 + e^-x))
logmodel = LogisticRegression()
#
# logmodel.fit(X_scaled, y)
#
# predictions = logmodel.predict(X_scaled)

# fit/train the model on the training data
logmodel.fit(X, y)

# get predictions for the trained model
predictions_train = logmodel.predict(X)

# import test data ************************************************
test_nba = pd.read_csv('solution_no_answer.csv')

# convert categorical classes into binary values to feed into the model
test_action_type = pd.get_dummies(test_nba['ACTION_TYPE'])
test_shot_type = pd.get_dummies(test_nba['SHOT_TYPE'])
test_shot_zone_area = pd.get_dummies(test_nba['SHOT_ZONE_AREA'])
test_shot_zone_basic = pd.get_dummies(test_nba['SHOT_ZONE_BASIC'])
test_shot_zone_range = pd.get_dummies(test_nba['SHOT_ZONE_RANGE'])

# get new dataframe for test data
test_nba = test_data_loaded(test_nba)

# add the new features
test_nba = pd.concat([test_nba, test_action_type, test_shot_type, test_shot_zone_area,
                      test_shot_zone_basic, test_shot_zone_range], axis=1)

# set prediction data (return view/copy with column(s) removed)
X_test = test_nba.drop('GAME_EVENT_ID', axis=1)

# generate predictions for the withheld test data
predictions_test = logmodel.predict(X_test)
# #***************************************************************
# # create submission for Kaggle
# submission = pd.DataFrame({"GAME_EVENT_ID": test_nba_2["GAME_EVENT_ID"], "SHOT_MADE_FLAG": predictions})
#
# # write submission to csv
# submission.to_csv("new_logreg_submission.csv", index=False) # do not save index values

# evaluate performance of the model
print(classification_report(y, predictions_test))
print("Accuracy: {}".format(accuracy_score(y, predictions_test))) # can alternatively use model.score(X, y)