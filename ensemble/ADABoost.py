import pandas as pd
import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree
import pylab as pl

class AdaBoostClassifier():
	def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
		'''
		:param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
							   If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
							   You can pass the object of the estimator class
		:param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
		'''
		self.models = []
		self.n_estimators = n_estimators
		self.alphas = []
		self.predictions = []
		self.X = None
		self.y = None

	def fit(self, X, y):
		"""
		Function to train and construct the AdaBoostClassifier
		Inputs:
		X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
		y: pd.Series with rows corresponding to output variable (shape of Y is N)
		"""
		# initilaize sample weights
		samples_no = X.shape[0]
		sample_wts = np.full(samples_no, (1/samples_no))

		self.X = X
		self.y = y
		self.used_estimators = 0
		X.reset_index(drop='True', inplace=True)
		y.reset_index(drop='True', inplace=True)
		
		# loop for each estimator
		for j in range(self.n_estimators):
			# define the model and train and predict with initial sample weights
			curr_model = DecisionTreeClassifier(max_depth=1)
			curr_model.fit(X, y, sample_weight=sample_wts)
			predict_y = curr_model.predict(X)
			self.models.append(curr_model)

			# error calculation in sample weights
			error = 0
			for m in range(len(predict_y)):
				error += sample_wts[m]*(predict_y[m]!=y[m])
			# print(error)

			# aplha m calculation
			alpham = (1/2)*(math.log((1-error)/(error)))
			self.alphas.append(alpham)
			self.predictions.append(predict_y)

			# update weights according to the predicted values
			for k in range(len(sample_wts)):
				if predict_y[k]==y[k]:
					sample_wts[k] = sample_wts[k]*np.exp(-1*alpham)
				else:
					sample_wts[k] = sample_wts[k]*np.exp(alpham)

			# normalize the weights
			self.used_estimators+=1
			sm = sum(sample_wts)
			for k in range(len(sample_wts)):
				sample_wts[k] = sample_wts[k]/sm

	def predict(self, X):
		"""
		Input:
		X: pd.DataFrame with rows as samples and columns as features
		Output:
		y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
		"""

		# final preditions
		final_prediction = []
		X.reset_index(drop='True', inplace = True)

		# iterate over each row sample
		for j in range(X.shape[0]):
			ans = 0
			# add after multiplying with corresponding alpha
			for k in range(len(self.alphas)):
				if (self.predictions[k][j]==0):
					ans += self.alphas[k]*(-1)
				else:
					ans += self.alphas[k]
			# apply signum function
			out = np.sign(ans)
			if (out==-1):
				final_prediction.append(0)
			else:
				final_prediction.append(1)
		return final_prediction

	def plot(self, naming):
		"""
		Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
		Creates two figures
		Figure 1 consists of 1 row and `n_estimators` columns
		The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
		Further, the scatter plot should have the marker size corresponnding to the weight of each point.

		Figure 2 should also create a decision surface by combining the individual estimators

		Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

		This function should return [fig1, fig2]
		"""

		# plot the tree structure for each estimator
		for j in range(self.n_estimators):
			ntree = self.models[j]
			text_representation = tree.export_text(ntree)
			print(text_representation)

		# contour or decision surface
		num_classes = 2			# number of labels
		colors_taken = "rb"
		step_size = 0.02

		plt.figure(1)
		for i in range(self.n_estimators):
			classifier = self.models[i]
			X,y = self.X, self.y

			# plot the decision boundary
			plt.subplot(2,3,i+1)

			# get the max and min values of the samples
			Min_x, Max_x = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
			Min_y, Max_y = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
			
			# grid creation
			xx, yy = np.meshgrid(np.arange(Min_x, Max_x, step_size), np.arange(Min_y, Max_y, step_size))
			plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

			# predict the points to get the label area
			output= classifier.predict(np.c_[xx.ravel(), yy.ravel()])
			output= output.reshape(xx.shape)
			cs = plt.contourf(xx, yy, output, cmap=plt.cm.RdBu)

			# plot the training points
			if naming=='iris':
				plt.xlabel("sepal width - 0")
				plt.ylabel("petal width - 1")
				names = ['Non-Virginica', 'Virginica']
			else:
				plt.xlabel("feature 0")
				plt.ylabel("feature 1")
				names = ['class 0', 'class 1']

			for j, color in zip(range(num_classes), colors_taken):
				idx = np.where(y == j)
				plt.scatter(X.iloc[idx][0], X.iloc[idx][1], c=color, label=names[j], cmap=plt.cm.RdBu, edgecolor='black', s=15)

		plt.legend(loc='lower right', borderpad=0, handletextpad=0)
		plt.axis("tight")
		plt.savefig('./images/ADABoost_estimators_surface'+ naming + '.png')

		# tree structure
		plt.figure(2)
		mdl = classifier.fit(X,y)
		plot_tree(mdl, filled=True)
		plt.savefig('./images/ADABoost'+ naming + 'tree.png')

		# combined forest creation
		plt.figure(3)
		plot_idx = 1
		plt.subplot(1, 1, plot_idx)
		Min_x, Max_x = X.loc[:][0].min() - 1, X.loc[:][0].max() + 1
		Min_y, Max_y = X.loc[:][1].min() - 1, X.iloc[:][1].max() + 1
		
		xx, yy = np.meshgrid(np.arange(Min_x, Max_x, step_size), np.arange(Min_y, Max_y, step_size))
		
		# take each estimator
		for tress in self.models:
			output= tress.predict(np.c_[xx.ravel(), yy.ravel()])
			output= output.reshape(xx.shape)
			cs = plt.contourf(xx, yy, output, cmap=plt.cm.RdYlBu,alpha=0.1)

		pl.axis("tight")
		# assign colors to each point according to the class
		for i, c in zip(range(num_classes), colors_taken):
			idx = np.where(y == i)
			plt.scatter(X.iloc[idx][0], X.iloc[idx][1], c=c, label='i',cmap=plt.cm.RdYlBu, edgecolor='black',)
		
		plt.axis("tight")
		plt.suptitle("Decision surfaces for ADABoost")
		plt.savefig('./images/ADABoost_combined_surfaces'+ naming+'.png')
		plt.show()