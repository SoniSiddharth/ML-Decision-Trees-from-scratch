import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor
from sklearn import datasets
###Write code here

# load iris from sklearn
iris_dataset = datasets.load_iris()
X = pd.DataFrame(iris_dataset.data)
y = pd.Series(iris_dataset.target)
l = len(X)
X = X.sample(frac=1.0)

# remove the attributes other than sepal width and petal width
X['label'] = y
X = X.drop([0], axis=1)
X = X.drop([2], axis=1)

# reindex the column names
X = X.rename(columns={1: 0, 3: 1})

# 60 percent of data for training and rest for testing
train_set = X.iloc[:int(l*0.6),:]
test_set = X.iloc[int(l*0.6):,:]

train_y = train_set['label']
test_y = test_set['label']

train_X = train_set.drop(['label'], axis=1)
test_X = test_set.drop(['label'], axis=1)

n_estimators = 3
Classifier = RandomForestClassifier(n_estimators=n_estimators,criterion='information_gain')
Classifier.fit(train_X, train_y)
y_hat = Classifier.predict(test_X)
Classifier.plot()
# [fig1, fig2] = Classifier_AB.plot()
print('Accuracy: ', accuracy(y_hat, test_y))
for cls in set(test_y):
    print('Precision: ', precision(y_hat, test_y, cls))
    print('Recall: ', recall(y_hat, test_y, cls))
