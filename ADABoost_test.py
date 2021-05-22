"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
# from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from sklearn import tree
from sklearn import datasets

np.random.seed(42)

pd.options.mode.chained_assignment = None

########### AdaBoostClassifier on Real Input and Discrete Output ###################
print('---------------------------------- PART 1 ----------------------------')

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
# [fig1, fig2] = Classifier_AB.plot()
Classifier_AB.plot('part1')
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))


# -------------------------- PART 2 ------------------------------------------
##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features

print('---------------------------------- PART 2 --------------------------------')

np.random.seed(42)

iris_dataset = datasets.load_iris()
X_iris = pd.DataFrame(iris_dataset.data)
y_iris = pd.Series(iris_dataset.target)
l = len(X_iris)
X_iris = X_iris.sample(frac=1.0)
y_iris = y_iris.replace(to_replace = 1, value = 0)
y_iris = y_iris.replace(to_replace = 2, value = 1)

X_iris['label'] = y_iris
X_iris = X_iris.drop([0], axis=1)
X_iris = X_iris.drop([2], axis=1)

X_iris = X_iris.rename(columns={1: 0, 3: 1})
train_set = X_iris.iloc[:int(l*0.6),:]
test_set = X_iris.iloc[int(l*0.6):,:]

train_y = train_set['label']
test_y = test_set['label']

train_X = train_set.drop(['label'], axis=1)
test_X = test_set.drop(['label'], axis=1)

np.random.seed(42)

n_estimators = 3
criteria = 'information_gain'
Classifier = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier.fit(train_X, train_y)
y_predict = Classifier.predict(test_X)
# [fig1, fig2] = Classifier.plot()
Classifier.plot('part2_iris')

print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_predict, test_y))
for cls in test_y.unique():
    print('Precision: ', precision(y_predict, test_y, cls))
    print('Recall: ', recall(y_predict, test_y, cls))