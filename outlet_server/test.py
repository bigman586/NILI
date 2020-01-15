import random
import time

import pandas as pd
from sklearn.ensemble import VotingClassifier

import graph
import model
import numpy as np
import db_setup
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

# splits = np.linspace(0.1, 1.0, 10, endpoint=True)
#
# model.load_data()
# model.prepare_data()
# model.load_models()
#
# start = time.time()
#
# names = ["DT", "KNN"]
# accuracies = []
#
# dt = model.dt
# knn = model.knn

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
# model.cross_val(model.knn)
# model.saveModels()
#
# dt = clf.best_estimator_
# model.cross_val(dt)
#
# model.dt = dt
# model.saveModels()

# accuracies.append(model.cross_val(model.dt))


# accuracies.append(model.cross_val(model.knn))

# graph.graphBar(names, accuracies)
# print("Program ran in " + str(time.time() - start) + " seconds")

# TODO: Add voting classifier and refine it
# model.cross_val(VotingClassifier(estimators=[('dt', dt), ('knn', knn)], voting='soft'))


model.load_models()
dt = model.dt
knn = model.knn

columns = model.get_columns()

df = pd.DataFrame()
df = df.append(
    {columns[0]: 0.151686, columns[1]: 0.18307, columns[2]: 0.0605465, columns[3]: 0.00366587, columns[4]: 0.13076,
     columns[5]: 0.07846, columns[6]: 0.07846, columns[7]: 0.20922}, ignore_index=True)

df = df[columns]

print(not np.any([0,0,0,0,0,]))
dt_dict = {"prediction": model.decode(dt.predict(df)[0])}
knn_dict = {"prediction": model.decode(knn.predict(df)[0])}

prediction = ""

if(dt_dict["prediction"] == knn_dict["prediction"]):
    prediction = dt_dict["prediction"] or knn_dict["prediction"]
else:
    roll = random.randint(1, 100)

    if roll <= 50:
        prediction = dt_dict["prediction"]
    elif roll >= 51:
        prediction = knn_dict["prediction"]

print(prediction)


