
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# RF classifier
# RandomForestClassifier
# importance



y = y.astype(np.float)
treeNum = 10000
survived_weight = .75
n_estimator = treeNum
oob_score = True
random_state =1
max_features = 25
y_weights = np.array([survived_weight if s == 0 else 1 for s in y])
forest = rfc(oob_score=True, n_estimators=treeNum)
forest.fit(va.T, y, sample_weight=y_weights)
importances = forest.feature_importances_
importances = 100.0 * (importances / importances.max())

sqrtfeat = int(np.sqrt(va.shape[1]))
minsampsplit = int(va.shape[0] * 0.015)
grid_test1 = {"n_estimators": [1000, 2500, 5000],
              "criterion": ["gini", "entropy"],
              "max_features": [sqrtfeat - 1, sqrtfeat, sqrtfeat + 1],
              "max_depth": [5, 10, 25],
              "min_samples_split": [2, 5, 10, minsampsplit]
              }
forest2 = rfc(oob_score=True)

print
"hp opt using GridSearchCV.."
grid_search = sl.grid_search.GridSearchCV(forest2, grid_test1, n_jobs=-1, cv=10)
grid_search.fit(va, y)

# best_params_from_gs = scorereport.report(grid_search, grid_search.score)
