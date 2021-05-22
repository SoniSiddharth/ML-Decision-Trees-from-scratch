import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn import datasets

np.random.seed(42)

# Read IRIS data set
# ...
# 

iris_dataset = datasets.load_iris()
X = pd.DataFrame(iris_dataset.data)
y = pd.Series(iris_dataset.target)
l = len(X)
X = X.sample(frac=1.0)
X['label'] = y
train_set = X.iloc[:int(l*0.7),:]
test_set = X.iloc[int(l*0.7):,:]

train_y = train_set['label']
test_y = test_set['label']

train_set = train_set.drop(['label'], axis=1)
test_set = test_set.drop(['label'], axis=1)

# -------------------------- PART 1 ---------------------------------------------

print("--------------- PART 1 --------------------------")

# print(train_set)
for criteria in ['information_gain', 'gini_index']:
	tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
	tree.fit(train_set, train_y)
	y_hat = tree.predict(test_set)
	tree.plot()
	print('Criteria :', criteria)
	print('Accuracy: ', accuracy(y_hat, test_y))
	for cls in test_y.unique():
		print('Precision: ', precision(y_hat, test_y, cls))
		print('Recall: ', recall(y_hat, test_y, cls))

# ---------------------------------- PART 2 -------------------------------------

print("--------------- PART 2 --------------------------")

def five_fold_cross_validation(X,y):
	l = X.shape[0]
	fold_accuracies = []
	depths = []

	# divide into test and train (outer loop)
	for t in range(5):
		print('------------------------- FOLD ', t+1, ' ---------------------------')
		X = X.sample(frac=1.0)
		fold_train = X.iloc[:int(l*0.8),:]
		fold_test = X.iloc[int(l*0.8):,:]
		arr = np.array_split(fold_train, 5)
		accuracy_arrs = []

		# divide each training set into train and validation set
		for j in range(5):
			val_data = arr[0]
			# print(val_data)
			val_y = val_data['label']
			val_X = val_data.drop(['label'], axis=1)
			frames = []
			for k in range(5):
				if (k!=j):
					frames.append(arr[k])
			train_data = pd.concat(frames)
			
			train_y = train_data['label']
			train_X = train_data.drop(['label'], axis=1)
			# print(train_X)
			# print(train_y)

			# train the model with different depths and test it on the validation set
			depth_acc = [0]		
			for m in range(1, 11):
				tree = DecisionTree('inforamation_gain', m)
				tree.fit(train_X, train_y)
				y_hat = tree.predict(val_X)
				acc = accuracy(y_hat, val_y)
				depth_acc.append(acc)
			accuracy_arrs.append(depth_acc)

		# calulate average accuracy for each depth
		l = len(accuracy_arrs)
		avg_accs = [0]
		for t in range(1,11):
			temp = []
			for j in range(5):
				temp.append(accuracy_arrs[j][t])
			avg = np.mean(temp)
			avg_accs.append(avg)
	
		# print(accuracy_arrs)
		# print(avg_accs)
		for k in range(1,11):
			if(avg_accs[k]>avg_accs[k-1]):
				print("depth: ", k, ' --> accuracy: ', avg_accs[k])
		
		# take the best depth from validation
		best_depth = 10
		max_acc = 0
		for k in range(1,11):
			if(avg_accs[k]>max_acc):
				max_acc = avg_accs[k]
				best_depth = k

		# train the model using best depth and calculate the accuracy on test set
		testing_y = fold_train['label']
		testing_X = fold_train.drop(['label'], axis=1)
		final_test_y = fold_test['label']
		final_test_X = fold_test.drop(['label'], axis=1)
		tree = DecisionTree('inforamation_gain', best_depth)
		tree.fit(testing_X, testing_y)
		y_final = tree.predict(final_test_X)
		ac = accuracy(y_final, final_test_y)
		fold_accuracies.append(ac)
		depths.append(best_depth)
	# print(fold_accuracies)
	return fold_accuracies, depths

accuracies, corress_depths = five_fold_cross_validation(X, y)
for j in range(5):
	print("Accuracy on test set is --> ", accuracies[j], " with depth --> ", corress_depths[j])
print("Average Accuracy --> ", np.mean(accuracies))