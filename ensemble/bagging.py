import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pylab as pl


class BaggingClassifier():
	def __init__(self, base_estimator, n_estimators=100):
		'''
		:param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
							   You can pass the object of the estimator class
		:param n_estimators: The number of estimators/models in ensemble.
		'''
		self.models = []
		self.n_estimators = n_estimators
		self.training = []
		self.X = None
		self.y = None

		for j in range(n_estimators):
			self.models.append(base_estimator.DecisionTreeClassifier(criterion='entropy'))

	def fit(self, X, y):
		"""
		Function to train and construct the BaggingClassifier
		Inputs:
		X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
		y: pd.Series with rows corresponding to output variable (shape of Y is N)
		"""
		self.X = X
		self.y = y
		X['y'] = y
		# print(X)
		
		# train n estimators 
		for j in range(self.n_estimators):
			current_data = X.sample(frac=1.0)		# data shuffling
			current_data = current_data.sample(frac=1.0, axis='rows', replace=True)		# extract rows with replacement
			# print(current_data)
			train_y = current_data['y']
			train_X = current_data.drop(['y'], axis=1)
			curr_tree = self.models[j]

			# train the current model
			curr_tree.fit(train_X, train_y)
			# keep track of train set used
			self.training.append((train_X, train_y))
		
		X = X.drop(['y'], axis=1)
		self.X = X

	def predict(self, X):
		"""
		Funtion to run the BaggingClassifier on a data point
		Input:
		X: pd.DataFrame with rows as samples and columns as features
		Output:
		y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
		"""
		predicted_y = []
		# print(X)
		for j in range(self.n_estimators):
			curr_tree = self.models[j]
			y_hat = curr_tree.predict(X)
			predicted_y.append(y_hat)
		# print(predicted_y)

		# take vote for each sample from the estimators
		final_prediction = []
		for j in range(X.shape[0]):
			temp = []
			for k in range(self.n_estimators):
				temp.append(predicted_y[k][j])
			temp = np.array(temp)
			final_prediction.append(np.bincount(temp).argmax())
		return final_prediction

	def plot(self):
		"""
		Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
		Creates two figures
		Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
		The title of each of the estimator should be iteration number

		Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

		Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

		This function should return [fig1, fig2]

		"""
		# tree structure plotting
		for j in range(self.n_estimators):
			ntree = self.models[j]
			text_representation = tree.export_text(ntree)
			print(text_representation)

		# if feature size is two plot the contour or decidion surface
		if self.X.shape[1]==2:
			self.plot_combined_surface()

	def plot_combined_surface(self):
		num_classes = len(set(self.y))		# number of classes
		colors_taken = "ryb"
		step_size = 0.02

		# figure 1 start
		plt.figure(1)
		for i in range(self.n_estimators):
			classifier = self.models[i]
			X,y = self.training[i]
			plt.subplot(2,3,i+1)		# decision boundary

			# extract min and max values of rows
			Min_x, Max_x = X.loc[:][0].min() - 1, X.loc[:][0].max() + 1
			Min_y, Max_y = X.loc[:][1].min() - 1, X.iloc[:][1].max() + 1
			xx, yy = np.meshgrid(np.arange(Min_x, Max_x, step_size), np.arange(Min_y, Max_y, step_size))
			plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

			# predict to create the contour area (colored)
			output= classifier.predict(np.c_[xx.ravel(), yy.ravel()])
			output= output.reshape(xx.shape)
			cs = plt.contourf(xx, yy, output, cmap=plt.cm.RdYlBu)

			plt.xlabel("X")
			plt.ylabel("y")

			# plot the training points - scattering
			for j, color in zip(range(num_classes), colors_taken):
				idx = np.where(y == j)
				plt.scatter(X.iloc[idx][0], X.iloc[idx][1], c=color, label=str(j), cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

		plt.suptitle("Decision surface of for Bagging")
		plt.legend(loc='lower right', borderpad=0, handletextpad=0)
		plt.axis("tight")
		plt.savefig('./images/Bagging_estimators.png')
		
		# combined decision surface - figure 2
		plt.figure(2)
		plot_idx = 1
		plt.subplot(1, 1, plot_idx)

		Min_x, Max_x = X.loc[:][0].min() - 1, X.loc[:][0].max() + 1
		Min_y, Max_y = X.loc[:][1].min() - 1, X.iloc[:][1].max() + 1
		
		xx, yy = np.meshgrid(np.arange(Min_x, Max_x, step_size), np.arange(Min_y, Max_y, step_size))

		# plot each estimator on same figure
		for tress in self.models:
			output= tress.predict(np.c_[xx.ravel(), yy.ravel()])
			output= output.reshape(xx.shape)
			cs = plt.contourf(xx, yy, output, cmap=plt.cm.RdYlBu,alpha=0.1)
		pl.axis("tight")
		
		# assign color to points based on class
		for i, c in zip(range(num_classes), colors_taken):
			idx = np.where(y == i)
			plt.scatter(X.iloc[idx][0], X.iloc[idx][1], c=c, label='i',cmap=plt.cm.RdYlBu, edgecolor='black',)

		plt.axis("tight")
		plt.suptitle("Combined Decision surfaces for Bagging")
		plt.savefig('./images/Bagging_combined_surfaces.png')
		plt.show()