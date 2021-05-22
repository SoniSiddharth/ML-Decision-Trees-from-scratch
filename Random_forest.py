"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)

########### RandomForestClassifier ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")

X['y'] = y
X = X.sample(frac=1.0)
y = X['y']
X = X.drop(['y'], axis=1)

for criteria in ['information_gain', 'gini_index']:
    Classifier = RandomForestClassifier(10, criterion = criteria)
    Classifier.fit(X, y)
    y_hat = Classifier.predict(X)
    Classifier.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

########### RandomForestRegressor ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

X['y'] = y
X = X.sample(frac=1.0)
y = X['y']
X = X.drop(['y'], axis=1)

Regressor = RandomForestRegressor(6)
Regressor.fit(X, y)
y_hat = Regressor.predict(X)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
