import pandas as pd
import sklearn as sl
import numpy as np
from sklearn import preprocessing
dtype=[('1',np.float64)]
train = pd.read_csv("../RF/Rawfeatures.csv",sep=',',header=-1)
labels = pd.read_csv("../RF/Rawlabel.csv",sep=',',header=-1)
types = list(train.dtypes)

train.select_dtypes(types)
ttypes=list(train.dtypes)
NanCheckMatrix = np.isnan(train).any()
print(NanCheckMatrix)
with open("..\RF\checkNaN.txt",'wb') as f:
    np.savetxt(f, NanCheckMatrix, fmt='%s', delimiter = ',')
scaler = preprocessing.StandardScaler().fit(train)
min_max_scaler = preprocessing.MinMaxScaler()
nomarlizedLables= min_max_scaler.fit_transform(train)
features = nomarlizedLables
features  = pd.DataFrame(features)
index = pd.DataFrame(labels[0],columns=["index"])
features["index"]=index["index"]
cols = list(features)
cols.insert(0, cols.pop(cols.index('index')))
features = features.ix[:, cols]
# o = np.ones(np.shape(features))
with open("..\RF\Features.csv",'wb') as f:
    np.savetxt(f, features, fmt='%s', delimiter = ',')