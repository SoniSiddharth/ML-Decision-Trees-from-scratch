help:
	@echo "make decision_tree : For running all four cases (Discrete and Real) of basic decision tree"
	@echo "make iris : For running the decision tree model on Iris dataset"
	@echo "make real_estate : For running the decison tree model on Real estate price prediction dataset"
	@echo "make experiments : For creating plots for time v/s Sample size of dataset"
	@echo "make adaboost : For testing the adaboost classifier"
	@echo "make bagging : For testing the Bagging model of Decision Trees"
	@echo "make random_forest : For testing the Random Forest model"
	@echo "make random_forest_iris : For running random forest model on Iris dataset"

decision_tree:
	@ python usage.py

iris:
	@ python iris-experiments.py

real_estate:
	@ python estate-experiments.py 

experiments:
	@ python experiments.py 

adaboost:
	@ python ADABoost_test.py 

bagging:
	@ python Bagging_test.py 

random_forest:
	@ python Random_forest.py 

random_forest_iris:
	@ python random_forest_iris.py 