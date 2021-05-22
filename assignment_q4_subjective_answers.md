# Question 4 Observations

The plots have been constructed by taking number of samples (rows) as X-axis and time taken to predict or training as Y-axis. Number of attributes (features) has been varied from 10 to 50. It is clearly visible that as the number of features increases keeping the number of columns constant time taken to train the model and to predict also increases.

- Prediction takes time almost proportional to number of features.

- The maximum height of the tree in discrete input case can be equal to the number of features.

- The maximum height of the tree in case of real input can be equal to the number of samples because each sample can possibly divide the dataset into parts in the worst condition.

- Time complexity to train a real input decision tree is **O(NMd)**, where N = number of features, M = number of samples, d = maximum depth of the tree

- In case of discrete input the time complexity is not trivial since it depends upon the number of discrete classes an attribute possess. Since we divide the node into each class and then for each child nodes, number of features is reduced by one. So the time complexity is **O(Mdd)**.
