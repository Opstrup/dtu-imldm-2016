import numpy as np
from sklearn import tree

from read_data import *

X = X.A
y = y.A.ravel()

# Fit regression tree classifier, Gini split criterion, no pruning
dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=40)
dtc = dtc.fit(X,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='DecisionTree.gvz', feature_names=attributeNames[4:-1])