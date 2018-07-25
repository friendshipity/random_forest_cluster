import pandas as pd
import sklearn as sl
import numpy as np
from sklearn import preprocessing
train = pd.read_csv("../RF/Rawlabel.csv",sep=',',header=-1)
labels = train.iloc[:,[1,3,4,5,6,7,8,9]]
NanCheckMatrix = np.isnan(train).any()
print(NanCheckMatrix)
print("123")
scaler = preprocessing.StandardScaler().fit(labels)
min_max_scaler = preprocessing.MinMaxScaler()
nomarlizedLables= min_max_scaler.fit_transform(labels)
labels = nomarlizedLables
o = np.ones(np.shape(labels))


for i,row in enumerate(labels):
    for j,element in enumerate(row):
        if element >= 0 and element < 0.20:
            labels[i][j] = 1
        if element >= 0.20 and element < 0.4:
            labels[i][j] = 2
        if element >= 0.4 and element <= 0.6:
            labels[i][j] = 3
        if element > 0.6 and element <= 0.8:
            labels[i][j] = 4
        if element > 0.8 and element <= 1:
            labels[i][j] = 5

labels  = pd.DataFrame(labels,columns=["a","b","c","d","e","f","g","h"])
index = pd.DataFrame(train[0],columns=["index"])
labels["index"]=index["index"]
cols = list(labels)
cols.insert(0, cols.pop(cols.index('index')))
labels = labels.ix[:, cols]


print("123")
with open("..\RF\Lables.csv",'wb') as f:
    np.savetxt(f, labels, fmt='%s', delimiter = ',')