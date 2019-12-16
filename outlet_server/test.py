import time
import graph
import model
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

splits = np.linspace(0.1, 1.0, 10, endpoint=True)

model.loadData()
model.prepareData()
model.loadModels()

start = time.time()

names = ["DT", "KNN"]
accuracies = []

# knn_grid_param = {
#     'n_neighbors': list(range(1, 10)),
#     'weights': ['uniform', 'distance'],
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     'leaf_size': list(range(1, 10)),
#     'p': list(range(1, 10))
# }
#
# dt_grid_param = {
#     'criterion': ['gini', 'entropy'],
#     'splitter': ['best', 'random'],
#     'max_depth': list(range(1, 10)),
#     'min_samples_split': list(range(2, 11)),
#     'min_samples_leaf': list(range(1, 10))
# }
#
# clf = GridSearchCV(estimator=DecisionTreeClassifier(),
#                    param_grid=dt_grid_param,
#                    scoring='accuracy',
#                    cv=5,
#                    n_jobs=-1)
#
#
# clf.fit(model.x_train, model.y_train)
#
# params = clf.best_params_
# print("Best parameters - " + str(params))
# print("Best score - " + str(clf.best_score_))
model.crossVal(model.dt)
#model.saveModels()
#
# dt = clf.best_estimator_
# model.crossVal(dt)
#
# model.dt = dt
# model.saveModels()

# accuracies.append(model.crossVal(model.dt))


# accuracies.append(model.crossVal(model.knn))

# graph.graphBar(names, accuracies)
# print("Program ran in " + str(time.time() - start) + " seconds")
