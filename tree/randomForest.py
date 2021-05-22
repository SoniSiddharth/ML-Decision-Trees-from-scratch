from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from tree.base import DecisionTree
from sklearn import tree
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.tree import plot_tree
import numpy as np
import pandas as pd
import math

class RandomForestClassifier():
	def __init__(self, n_estimators=100, criterion='gini', max_depth=10):
		'''
		:param estimators: DecisionTree
		:param n_estimators: The number of trees in the forest.
		:param criterion: The function to measure the quality of a split.
		:param max_depth: The maximum depth of the tree.
		'''
		self.trees = []
		self.n_estimators = n_estimators
		self.criterion = criterion
		self.training = []
		self.max_depth = max_depth
		self.X = None
		self.y = None
		self.attributes = []

		for j in range(n_estimators):
			d_tree = DecisionTreeClassifier(max_depth=10, criterion='entropy')
			self.trees.append(d_tree)

	def fit(self, X, y):
		"""
		Function to train and construct the RandomForestClassifier
		Inputs:
		X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
		y: pd.Series with rows corresponding to output variable (shape of Y is N)
		"""
		self.X = X
		self.y = y
		for i in range(self.n_estimators):
			tree = self.trees[i]
			if (X.shape[1]<=2):
				curr_X = X.sample(frac=1.0)
			else:
				curr_X = X.sample(int(math.sqrt(X.shape[1])),axis=1) # extract random columns
			self.attributes.append(list(curr_X.columns))		# keep track of train sets
			curr_y = y[curr_X.index]							# modify y similarly
			self.training.append((curr_X.copy(), curr_y.copy()))
			tree.fit(curr_X, curr_y)

	def predict(self, X):
		"""
		Funtion to run the RandomForestClassifier on a data point
		Input:
		X: pd.DataFrame with rows as samples and columns as features
		Output:
		y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
		"""
		# prediction by individual estimator
		predictions = []
		# supply same attributes which were trained to that model
		for i in range(self.n_estimators):
			predictions.append(self.trees[i].predict(X[self.attributes[i]]))

		# iterate over the samples
		final_prediction = []
		for j in range(X.shape[0]):
			temp = []
			for k in range(self.n_estimators):
				temp.append(predictions[k][j])
			temp = np.array(temp)
			# take mode of the individual predictions
			final_prediction.append(np.bincount(temp).argmax())
		return final_prediction

	def plot(self):
		"""
		Function to plot for the RandomForestClassifier.
		It creates three figures

		1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
		If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

		2. Creates a figure showing the decision surface for each estimator

		3. Creates a figure showing the combined decision surface

		"""
		# plot the tree structure of each estimator
		for j in range(self.n_estimators):
			ntree = self.trees[j]
			text_representation = tree.export_text(ntree)
			print(text_representation)
		
		# decision surface plots if features are 2
		if (self.X.shape[1]==2):
			self.plot_combined_surface()
		return

	# function to plot the decion surfaces
	def plot_combined_surface(self):
		num_classes = len(set(self.y))		# number of labels
		colors_taken = "ryb"
		step_size = 0.02

		plt.figure(1)
		for i in range(self.n_estimators):
			classifier = self.trees[i]
			X,y = self.training[i]
			plt.subplot(2,3,i+1)		# decision boundary

			# extract min and max values of rows
			Min_x, Max_x = X.loc[:][0].min() - 1, X.loc[:][0].max() + 1
			Min_y, Max_y = X.loc[:][1].min() - 1, X.iloc[:][1].max() + 1

			xx, yy = np.meshgrid(np.arange(Min_x, Max_x, step_size), np.arange(Min_y, Max_y, step_size))
			plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

			# plot the contour or decision area (colored)
			output= classifier.predict(np.c_[xx.ravel(), yy.ravel()])
			output= output.reshape(xx.shape)
			cs = plt.contourf(xx, yy, output, cmap=plt.cm.RdYlBu)

			plt.xlabel("X")
			plt.ylabel("y")

			# plt the points based on class
			for j, color in zip(range(num_classes), colors_taken):
				idx = np.where(y == j)
				plt.scatter(X.iloc[idx][0], X.iloc[idx][1], c=color, label=str(j), cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

		plt.suptitle("Decision surface of a decision tree using paired features")
		plt.legend(loc='lower right', borderpad=0, handletextpad=0)
		plt.axis("tight")
		plt.savefig('./images/random_forest_individual.png')
		
		# figure 2 started
		plt.figure(2)
		plot_idx = 1
		plt.subplot(1, 1, plot_idx)

		# extract min and max values of rows
		Min_x, Max_x = X.loc[:][0].min() - 1, X.loc[:][0].max() + 1
		Min_y, Max_y = X.loc[:][1].min() - 1, X.iloc[:][1].max() + 1
		
		xx, yy = np.meshgrid(np.arange(Min_x, Max_x, step_size), np.arange(Min_y, Max_y, step_size))

		for t in self.trees:
			output= t.predict(np.c_[xx.ravel(), yy.ravel()])
			output= output.reshape(xx.shape)
			cs = plt.contourf(xx, yy, output, cmap=plt.cm.RdYlBu,alpha=0.1)
		pl.axis("tight")

		# assign color to points based on class
		for i, c in zip(range(num_classes), colors_taken):
			idx = np.where(y == i)
			plt.scatter(X.iloc[idx][0], X.iloc[idx][1], c=c, label='i',cmap=plt.cm.RdYlBu, edgecolor='black',)

		plt.axis("tight")
		plt.suptitle("Decision surfaces for Random forest")
		plt.savefig('./images/random_forest.png')

		# plot the tree structure the last estimator
		plt.figure(3)
		mdl = classifier.fit(X,y)
		plot_tree(mdl, filled=True)
		plt.savefig('./images/Random_forest_tree_structure.png')	
		plt.show()

class RandomForestRegressor():
	def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
		'''
		:param n_estimators: The number of trees in the forest.
		:param criterion: The function to measure the quality of a split.
		:param max_depth: The maximum depth of the tree.
		'''
		self.trees = []
		self.n_estimators = n_estimators
		self.criterion = criterion
		self.X = None
		self.y = None
		self.attributes = []
		self.training = []

		for j in range(n_estimators):
			d_tree = DecisionTreeRegressor()
			self.trees.append(d_tree)

	def fit(self, X, y):
		"""
		Function to train and construct the RandomForestRegressor
		Inputs:
		X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
		y: pd.Series with rows corresponding to output variable (shape of Y is N)
		"""
		self.X = X
		self.y = y

		# train each estimtor with reduced samples
		for i in range(self.n_estimators):
			tree = self.trees[i]

			# take some features randomly
			curr_X = X.sample(int(math.sqrt(X.shape[1])),axis=1)
			# get the columns remained
			self.attributes.append(list(curr_X.columns))
			curr_y = y[curr_X.index]
			# train the model
			tree.fit(curr_X, curr_y)
			# keep track of train sets
			self.training.append((curr_X.copy(), curr_y.copy()))

	def predict(self, X):
		"""
		Funtion to run the RandomForestRegressor on a data point
		Input:
		X: pd.DataFrame with rows as samples and columns as features
		Output:
		y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
		"""
		# predictions
		predictions = []
		for i in range(self.n_estimators):
			predictions.append(self.trees[i].predict(X[self.attributes[i]]))

		# take the mean of values got from each estimator
		final_prediction = []
		# iterate over the samples
		for j in range(X.shape[0]):
			temp = []
			for k in range(self.n_estimators):
				temp.append(predictions[k][j])
			temp = np.array(temp)
			final_prediction.append(np.mean(temp))
		return final_prediction

	def plot(self):
		"""
		Function to plot for the RandomForestClassifier.
		It creates three figures

		1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
		If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

		2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

		3. Creates a figure showing the combined decision surface/prediction

		"""
		for j in range(self.n_estimators):
			ntree = self.trees[j]
			text_representation = tree.export_text(ntree)
			print(text_representation)

	