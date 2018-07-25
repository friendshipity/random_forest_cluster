
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.datasets import load_iris
train = pd.read_csv("../RF/Rawfeatures.csv",sep=',',header=-1)
labels = pd.read_csv("../RF/Rawlabel.csv",sep=',',header=-1)
iris = load_iris()

clf =tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
