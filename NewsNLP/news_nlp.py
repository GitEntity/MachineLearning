# Author: Devante Wilson
# Date: April 10, 2018
#
# This program is a supervised learning classifier for news article popularity

# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk # tool kit for natural language processing

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# load the training data (labels)
labelsTrain = pd.read_csv('ctrain.csv', header=None, squeeze=True)
# load the training data (contents)
contentsTrain = pd.read_csv('training.csv', header=None)
# load the training data (titles)
titlesTrain = pd.read_csv('training2.csv', header=None)
# load the testing data (contents)
contentsTest = pd.read_csv('test.csv', header=None)
# load the testing data (titles)
titlesTest = pd.read_csv('test2.csv', header=None)

# create a classification model
clf = KNeighborsClassifier(n_jobs= -1)

# store the document-term matrix (X) and target vector (y)
X = contentsTrain
y = labelsTrain.values.ravel()

# fit/train the model on the training data
myModel = clf.fit(X, y)

# generate predictions for training data
predictionsTrain = myModel.predict(X)

# evaluate performance of the model
print('########## Training/Fitting report ##########')
print(classification_report(y, predictionsTrain))
print("Accuracy: {}".format(accuracy_score(y, predictionsTrain))) # can alternatively use myModel.score(X, y)

# # print the confusion matrix
# print('###### Confusion Matrix from Training #######')
# print(metrics.confusion_matrix(y, predictionsTrain))

# store the document-term matrix for the test data
X_test = contentsTest

# generate predictions for test data
predictionsTest = myModel.predict(X_test)

# generate id for .csv submission file
indexId = []
for i in range(1, len(predictionsTest) + 1):
    indexId.append(i)

# create submission for Kaggle
submission = pd.DataFrame({"id": indexId, "class": predictionsTest})

# write submission to .csv file
submission.to_csv("News_Submission_1.csv", index=False) # do not save index values
