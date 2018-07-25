import numpy as np
import csv
from ranking_SVM.baseline import baseline
from ranking_SVM.r_train import r_train
from ranking_SVM.r_predict import r_predict
from ranking_SVM.pca import pca

# data
train=np.arange(92)
test=np.arange(88)

x=[]
for t in csv.reader(open('x.csv', 'r')):
   x.append(t)
x=np.asarray(x,'f')
#x=pca(x)

y=[]
for t in csv.reader(open('y.csv', 'r')):
   y.append(t)
y=np.asarray(y,'d')

# train
rsvm=r_train(x[train],y[train])
# rank
r=r_predict(rsvm,x[test])
#r=baseline(x[test],x[-4:])

import pylab as pl
pl.scatter(r[:,0],r[:,1])
pl.plot([0,len(r)],[r[4,1],r[4,1]],'k--',lw=2)
pl.xlabel('CANDIDATE_ID')
pl.ylabel('SCORE')
pl.show()
