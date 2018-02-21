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
    data_frame = nba.drop(['ACTION_TYPE', 'EVENTTIME', 'EVENT_TYPE', 'GAME_DATE', 'GAME_ID', 'HTM',
                'MINUTES_REMAINING', 'PERIOD', 'PLAYER_ID', 'PLAYER_NAME', 'QUARTER', 'SECONDS_REMAINING',
                'SHOT_ATTEMPTED_FLAG', 'SHOT_TIME', 'SHOT_TYPE', 'SHOT_ZONE_AREA', 'SHOT_ZONE_BASIC',
                'SHOT_ZONE_RANGE', 'TEAM_ID', 'TEAM_NAME', 'VTM', 'LOC_X', 'LOC_Y'], axis=1)
    return data_frame

# define file name
file_name = 'solution_no_answer.csv'
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
event_type = pd.get_dummies(nba['EVENT_TYPE'], drop_first=True) # returns 'Missed Shot' as 1 (Made Shot=0)
shot_type = pd.get_dummies(nba['SHOT_TYPE'])
shot_zone_area = pd.get_dummies(nba['SHOT_ZONE_AREA'])
shot_zone_basic = pd.get_dummies(nba['SHOT_ZONE_BASIC'])
shot_zone_range = pd.get_dummies(nba['SHOT_ZONE_RANGE'])

# detect use of test set
if file_name=='solution_no_answer.csv':
    nba = test_data_loaded(nba)
else: # training or validation set are being used
    # remove old features
    nba = nba.drop(['ACTION_TYPE', 'EVENTTIME', 'EVENT_TYPE', 'GAME_DATE', 'GAME_ID', 'HTM',
                    'MINUTES_REMAINING', 'PERIOD', 'PLAYER_ID', 'PLAYER_NAME', 'QUARTER', 'SECONDS_REMAINING',
                    'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG', 'SHOT_TIME', 'SHOT_TYPE', 'SHOT_ZONE_AREA', 'SHOT_ZONE_BASIC',
                    'SHOT_ZONE_RANGE', 'TEAM_ID', 'TEAM_NAME', 'VTM', 'LOC_X', 'LOC_Y'], axis=1)

# add all of the new features
nba = pd.concat([nba, action_type, event_type, shot_type, shot_zone_area, shot_zone_basic, shot_zone_range], axis=1)

# define prediction target
y = nba['Missed Shot']
# set prediction data (return view/copy with column(s) removed)
X = nba.drop(['Missed Shot', 'GAME_EVENT_ID'], axis=1)
# create view that still includes 'GAME_EVENT_ID' header to merge into submission
nba_2 = nba.drop('Missed Shot', axis=1)

# create logistic regression model (transformed = 1 / (1 + e^-x))
logmodel = LogisticRegression()

# fit/train the model on the training data
logmodel.fit(X, y)

# generate predictions for the withheld test data
predictions = logmodel.predict(X)

# create submission for Kaggle
submission = pd.DataFrame({"GAME_EVENT_ID": nba_2["GAME_EVENT_ID"], "SHOT_MADE_FLAG": predictions})

# write submission to csv
submission.to_csv("logreg_submission.csv", index=False) # do not save index values

# evaluate performance of the model
print(classification_report(y, predictions))
print("Accuracy: {}".format(accuracy_score(y, predictions))) # can alternatively use model.score(X, y)

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