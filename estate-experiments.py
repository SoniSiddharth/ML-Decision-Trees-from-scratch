
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Read real-estate data set
# ...
# 
data = pd.DataFrame(pd.read_csv('realestate.csv', sep = '\t'))
arr = list(data.columns)
data = data.drop([arr[0]], axis=1)
for i in range(len(arr)-1):
	data = data.rename(columns={arr[i+1]: i})
# print(data)
train_y = data[data.shape[1]-1]
train_X = data.drop([data.shape[1]-1], axis=1)
# print(train_X)

# ------------------------------ PART 1 ---------------------------------------------

print('------------------------------ PART 1 ---------------------------------')

for criteria in ['information_gain', 'gini_index']:
	tree = DecisionTree(criterion=criteria, max_depth=10) #Split based on Inf. Gain
	tree.fit(train_X, train_y)
	y_hat = tree.predict(train_X)
	# print(y_hat)
	# print(train_y)
	print('Criteria :', criteria)
	print('RMSE: ', rmse(y_hat, train_y))
	print('MAE: ', mae(y_hat, train_y))

# ------------------------------ PART 2 ---------------------------------------------

print('------------------------------ PART 2 ---------------------------------')

sk_tree = DecisionTreeRegressor(max_depth=10)
sk_tree.fit(train_X, train_y)
y_hat = sk_tree.predict(train_X)
print('RMSE: ', rmse(y_hat, train_y))
print('MAE: ', mae(y_hat, train_y))
