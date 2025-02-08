import numpy as np
import pydotplus
from IPython.display import Image
from sklearn import tree

features = [[1,3],[2,3],[3,1],[3,1],[2,3]] # 1 for green, 2 for yellow, 3 for red
labels = [1,1,2,2,3] # 1 for apple, 2 for grape, 3 for lemon

clf = tree.DecisionTreeClassifier()
model = clf.fit(features, labels)

print(labels)
print(clf.predict(features))

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['color', 'diameter'], class_names=['apple', 'grape', 'lemon'], filled=True, rounded=True, impurity=False)
print(dot_data)

