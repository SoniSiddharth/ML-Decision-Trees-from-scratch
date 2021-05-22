"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index, gain_gini_index, variance

np.random.seed(42)

pd.options.mode.chained_assignment = None

# defining node for discrete input cases
class Node:
    def __init__(self):
        self.data = None        # label assosiated with the node 
        self.attr = None        # attibute used during the node
        self.children = dict()  # children nodes
        self.check = False      # check whether node is a leaf

# defining node for real input cases
class NodeReal:
    def __init__(self):
        self.data = None        # label assosiated with the node
        self.attr = None        # attibute used during the node
        self.left = None        # left child
        self.right = None       # right child
        self.check = False      # check whether node is a leaf

class DecisionTree():
    def __init__(self, criterion, max_depth=10):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.tree = None
        self.X = None
        self.y = None
        self.depth = 0
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y

        # check which type of data is used for training
        num = X.values[0][0]

        # discrete input case
        if type(num)==np.int64 or type(num)==np.int32 or type(num)==int:
            self.tree = Node()
            self.tree = self.discrete_input(X, y)
        else:
            # real input
            self.tree = NodeReal()
            num = y.values[0]
            # real input discrete output (RIDO)
            if type(num)==np.int64 or type(num)==np.int32 or type(num)==int:
                self.tree = self.real_input_discrete_output(self.tree, self.X, self.y, self.depth)
            else:
                # real input real output (RIRO)
                self.tree = self.real_input_real_output(self.tree, self.X, self.y, 0)

    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        num = X.values[0][0]
        # print(num)
        if type(num)==np.int64 or type(num)==np.int32 or type(num)==int:
            return self.discrete_predict(X)
        else:
            # print("hello")
            return self.real_predict(X)

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        num = self.X.values[0][0]
        if type(num)==np.int64 or type(num)==np.int32 or type(num)==int:
            self.plot_discrete(self.tree, 1)
        else:
            ntree = self.tree
            space = "   "
            self.plot_real_input(ntree, space)
    
    def plot_real_input(self, head, space):
        attr = head.attr
        val = head.data
        print("? (feature {} > {})".format(attr, val))
        print(" {} {} : ".format(space,'Y'), end='')
        if head.left.check:
            print("--> Class {}".format(head.left.data))
        else:
            self.plot_real_input(head.left,space+"   ")
        print(" {} {} : ".format(space,'N'), end='')
        if head.right.check:
            print("--> Class {}".format(head.right.data))
        else:
            self.plot_real_input(head.right, space+"   ")
        return

    def plot_discrete(self, head, space):
        if head.check:
            print('Class--> '+str(head.data))
            return
        val = head.attr
        print("? {})".format(val))
        space += 1
        for j in head.children:
            print("{} --> value {} ".format("   "*space, str(j)), end='')
            self.plot_discrete(head.children[j], space+4)

    def discrete_predict(self, X):
        predicted = []          # array to store predictions

        # iterate over rows
        for _,rw in X.iterrows():
            head = self.tree    # start node            
            
            # check for the leaf node
            while head.check==False:
                attr = head.attr
                value = rw[attr]
                if value in head.children:
                    head = head.children[value]     # go to the next child
                else:
                    num = self.y.values[0]

                    # DIDO, DIRO, RIDO, RIRO cases
                    if type(num)==np.int64 or type(num)==np.int32 or type(num)==int:
                        y = np.array(self.y)
                        vals, counts = np.unique(y, return_counts=True)
                        return (vals[np.argmax(counts)])        # taking mode of the data
                    else:
                        y = np.array(self.y)
                        return (sum(y)/y.size)                  # mean in case of real output
            predicted.append(head.data)
        return predicted

    def real_predict(self, X):
        predicted = []      # array to store the predictions
        for _,row in X.iterrows():
            head = self.tree        # start node

            # check for the leaf node
            while head.check==False:
                label = head.attr
                val = row[label]
                if val<=head.data:
                    head = head.left    # go the left node
                else:
                    head = head.right   # value is greater so choose right node
            # print(head.data)
            predicted.append(head.data)     # leaf node means end of traversal
        return predicted

    def max_infogain_attribute(self, dataset, labels):
            # current_labels = list(labels)
            max_entropy = -1 * 10**9
            chosen_attribute = 0        # attribute to be chosen based on max info gain
            if self.criterion=="information_gain":
                for j in dataset:
                    temp_entropy = information_gain(labels, dataset[j])     # entropy calculation
                    if temp_entropy>max_entropy:
                        max_entropy = temp_entropy
                        chosen_attribute = j
            else:
                # gini index criteria
                for j in dataset:
                    temp_entropy = gain_gini_index(labels, dataset[j])
                    if temp_entropy>max_entropy:
                        max_entropy = temp_entropy
                        chosen_attribute = j
            return chosen_attribute
    
    # function for discrete input (both cases DIDO and DIRO)
    def discrete_input(self, dataset, labels):
        node = Node()           # initialize node
        y_values = list(labels)
        unique_values = set(y_values)

        # if all labels are same
        if len(unique_values) == 1:
            node.data = y_values[0]
            node.check = True
            return node

        # if all attributes are used
        if dataset.shape[1] == 0:
            node.check = True
            num = labels.values[0]
            if type(num)==np.int64 or type(num)==np.int32 or type(num)==int:
                node.data = max(set(y_values), key=y_values.count)
            else:
                node.data = np.mean(y_values)
            return node

        # best attribute to split
        attribute_taken = self.max_infogain_attribute(dataset, labels)
        node.attr = attribute_taken

        attribute_values = set()
        for _,sample in dataset.iterrows():
            attribute_values.add(sample[attribute_taken])
        attribute_values = list(attribute_values)

        # print(attribute_taken)
        # each value of chosen attribute becomes a node
        for val in attribute_values:
            # print(val)
            con = dataset[attribute_taken]==val
            new_dataset = dataset[con].copy()
            new_dataset.drop([attribute_taken], axis=1, inplace=True)
            new_labels = labels[con]
            # print(new_dataset)
            # print(dataset)
            # if dataset has no samples it is a leaf node
            if new_dataset.shape[0]==0:
                child = Node()
                num = labels.values[0]
                node.children[val] = child
                child.check = True
                if type(num)==np.int64 or type(num)==np.int32 or type(num)==int:
                    child.data = max(set(labels), key=y_values.count)       # discrete output
                else:
                    child.data = np.mean(labels)        # real output
            else:
                # next node definition
                node.children[val] = self.discrete_input(new_dataset, new_labels)
        return node
    
    # mode of the data
    def modeofdata(self, arr):
        arr = np.array(arr)
        return np.bincount(arr).argmax()

    # helper function for real input training
    def get_division_point(self, arr, labels):
        l = labels.shape[0]
        max_gain = -1*(10**9)
        point = 0
        num = labels.values[0]

        # iterate over all the values of an attribute to find the best split it gives
        if type(num)==np.int64 or type(num)==np.int32 or type(num)==int:
            overall_entropy = entropy(labels)
            overall_gini = gini_index(labels)
        else:
            overall_entropy = variance(labels)
            overall_gini = variance(labels)
        
        # divide the labels based on attribute value
        for m in range(l):
            temp = arr[m][0]
            left_arr = []
            right_arr = []
            for n in range(l):
                if (arr[n][0]<=temp):
                    left_arr.append(arr[n][1])
                else:
                    right_arr.append(arr[n][1])
            # maximize the information gain
            if self.criterion=="information_gain":
                if type(num)==np.int64 or type(num)==np.int32 or type(num)==int:
                    le = (len(left_arr)/l)*entropy(left_arr)
                    re = (len(right_arr)/l)*entropy(right_arr)
                    eff_entropy = overall_entropy - le - re
                else:
                    le = (len(left_arr)/l)*variance(left_arr)
                    re = (len(right_arr)/l)*variance(right_arr)
                    eff_entropy = overall_entropy - le - re
            else:
                if type(num)==np.int64 or type(num)==np.int32 or type(num)==int:
                    le = (len(left_arr)/l)*gini_index(left_arr)
                    re = (len(right_arr)/l)*gini_index(right_arr)
                    eff_entropy = overall_gini - le -re
                else:
                    le = (len(left_arr)/l)*variance(left_arr)
                    re = (len(right_arr)/l)*variance(right_arr)
                    eff_entropy = overall_gini - le -re
            if eff_entropy>max_gain:
                max_gain = eff_entropy
                point = temp
        return point, max_gain

    def max_infogain_attribute_real_input(self, dataset, labels):
        attr_gain = None
        chosen_attribute = None
        pointbreak = None
        label_list = list(labels)

        # checking all the attributes for the best split
        for j in dataset:
            lst = list(dataset[j][:])
            # print(lst)
            combined_lst = []
            for k in range(len(lst)):
                combined_lst.append((lst[k], label_list[k]))
            combined_lst.sort(key=lambda x:x[0])
            # print(combined_lst)

            # sort the array to get the 2 sets
            division_value, temp_gain = self.get_division_point(combined_lst, labels)
            if attr_gain==None or temp_gain>attr_gain:
                attr_gain = temp_gain
                pointbreak = division_value
                chosen_attribute = j
        # print(chosen_attribute, pointbreak)

        # return the chosen attribute index and point of split
        return chosen_attribute, pointbreak

    def real_input_discrete_output(self, node, dataset, labels, depth):
        if not node:
            node = NodeReal()
        y_values = list(labels)

        unique_vals = np.array(labels.unique())

        # base conditions for leaf node
        if unique_vals.size==1:
            node.data = unique_vals[0]
            node.check = True
            return node
        if dataset.shape[1]==0:
            node.check = True
            node.data = self.modeofdata(y_values)
            return node

        # if depth exceeds max depth assign make it the leaf node
        if (depth >= self.max_depth):
            node.check = True
            node.data = self.modeofdata(y_values)
            return node
        else:
            # get the best attribute
            attribute_taken, pointbreak = self.max_infogain_attribute_real_input(dataset, labels)
            # print(type(dataset))
            con = dataset[attribute_taken]<=pointbreak
            left_dataset = dataset[con]
            left_labels = labels[con]
            con2 = dataset[attribute_taken]>pointbreak
            right_dataset = dataset[con2]
            right_labels = labels[con2]

            # divide the dataset cased on the condition
            node.data = pointbreak
            node.attr = attribute_taken

            # if division makes a side empty it is a leaf node
            if left_dataset.shape[0]==0:
                child = NodeReal()
                node.left = child
                node.left.check = True
                node.left.data = self.modeofdata(y_values)
            else:
                child = NodeReal()
                node.left = child
                node.left = self.real_input_discrete_output(node.left, left_dataset, left_labels, depth+1)

            if right_dataset.shape[0]==0:
                child = NodeReal()
                node.right = child
                node.left.check = True
                node.left.data = self.modeofdata(y_values)
            else:
                child = NodeReal()
                node.right = child
                node.right = self.real_input_discrete_output(node.right, right_dataset, right_labels, depth+1)
        return node
    
    def real_input_real_output(self, node, dataset, labels, depth):
        if not node:
            node = NodeReal()
        if dataset.shape[1]==0:
            node.check = True
            node.data = np.mean(labels)
            # print("leaf1")
            return node
        if (depth >= self.max_depth):
            node.check = True
            node.data = np.mean(labels)
            return node
        else:
            attribute_taken, pointbreak = self.max_infogain_attribute_real_input(dataset, labels)
            # print(attribute_taken)
            # print(pointbreak)
            # print(dataset)
            con = dataset[attribute_taken]<=pointbreak
            left_dataset = dataset[con]
            left_labels = labels[con]
            con2 = dataset[attribute_taken]>pointbreak
            right_dataset = dataset[con2]
            right_labels = labels[con2]

            node.data = pointbreak
            node.attr = attribute_taken
            if left_dataset.shape[0]==0:
                child = NodeReal()
                node.left = child
                node.left.data = np.mean(labels)
                node.left.check = True
                # print("leaf3")
            else:
                child = NodeReal()
                node.left = child
                node.left = self.real_input_real_output(node.left, left_dataset, left_labels, depth+1)

            if right_dataset.shape[0]==0:
                child = NodeReal()
                node.right = child
                node.right.data = np.mean(labels)
                node.right.check = True
                # print("leaf4")
            else:
                child = NodeReal()
                node.right = child
                node.right = self.real_input_real_output(node.right, right_dataset, right_labels, depth+1)
        return node