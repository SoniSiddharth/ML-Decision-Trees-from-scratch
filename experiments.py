
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

# time calculation function
def time_calculate(M, N, criteria):
    train_time = []
    predict_time = []
    for k in range(2,M):
        time_t = []
        time_p = []

        # fake dataset generation for each case
        for j in range(10,N,5):
            if (criteria=='DIDO'):
                X = pd.DataFrame({q:pd.Series(np.random.randint(2, size = j), dtype="category") for q in range(k)})
                y = pd.Series(np.random.randint(2, size = j), dtype="category")
            elif(criteria=='DIRO'):
                X = pd.DataFrame({q:pd.Series(np.random.randint(2, size = j), dtype="category") for q in range(k)})
                y = pd.Series(np.random.randn(j))
            elif(criteria=='RIDO'):
                X = pd.DataFrame({q:pd.Series(np.random.randn(j)) for q in range(k)})
                y = pd.Series(np.random.randint(2, size = j), dtype="category")
            else:
                X = pd.DataFrame({q:pd.Series(np.random.randn(j)) for q in range(k)})
                y = pd.Series(np.random.randn(j))
            
            # now train and predict to calculate time consumed during these
            start = time.time()
            tree = DecisionTree(criterion='information_gain')
            tree.fit(X,y)
            time_t.append(time.time() - start)
            start = time.time()
            yhat = tree.predict(X)
            time_p.append(time.time() - start)
            # print(yhat)
        train_time.append(time_t)
        predict_time.append(time_p)
    return train_time, predict_time

# plotting function
def graph_plot(x_axis, y_axis, name):
    for j in range(len(y_axis)):
        plt.plot(x_axis, y_axis[j][:], label=str(j+2)+" attributes")
    plt.xlabel('Number of Samples')
    plt.ylabel('Time Taken')
    plt.legend()
    plt.savefig('./images/experiments_'+name+'.png')
    plt.show()

# print(train_time)
# print(predict_time)


N = 50          # number of samples (rows)
M = 7           # number odf features (columns)
cases = ['DIDO', 'DIRO', 'RIDO', 'RIRO']        # four cases
for j in range(4):
    x_axis = [i for i in range(10,N,5)]
    training, predicting = time_calculate(M, N, cases[j])
    graph_plot(x_axis, training, 'train_'+cases[j])
    graph_plot(x_axis, predicting, 'predict_'+cases[j])