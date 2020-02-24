import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import db_setup
import graph
import model


# Utility function to report best scores
def report(results, n_top=3):
    parameters = []
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            parameters.append(results['params'][candidate])
            print("")

    return parameters


knn_grid_param = {
    'n_neighbors': list(range(1, 50)),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': list(range(1, 50)),
    'p': list(range(1, 50))
}

dt_grid_param = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': list(range(1, 50)),
    'min_samples_split': list(range(2, 50)),
    'min_samples_leaf': list(range(1, 50))
}


def prepare():
    model.load_data()
    model.prepare_data()
    model.load_models()


def grid_optimize():
    prepare()

    clf_knn = GridSearchCV(estimator=KNeighborsClassifier(),
                           param_grid=knn_grid_param,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1)

    clf_dt = GridSearchCV(estimator=DecisionTreeClassifier(),
                          param_grid=dt_grid_param,
                          scoring='accuracy',
                          cv=5,
                          n_jobs=-1)

    print(model.dt)
    clf_knn.fit(model.x_train, model.y_train)
    clf_dt.fit(model.x_train, model.y_train)

    # params_dt = clf_dt.best_params_
    # params_knn = clf_knn.best_params_

    print("Best parameters - " + str(params))
    print("Best score - " + str(clf.best_score_))

    model.knn = clf_knn.best_estimator_
    model.dt = clf_dt.best_estimator_

    model.save_models()


def random_optimize():
    prepare()

    # run randomized search
    n_iter_search = 100
    random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=dt_grid_param,
                                       n_iter=n_iter_search)

    start = time.time()
    random_search.fit(model.x_train, model.y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))

    results = report(random_search.cv_results_)
    for result in results:
        print(model.cross_val_model(
            DecisionTreeClassifier(splitter=result['splitter'], min_samples_split=result['min_samples_split'],
                                   min_samples_leaf=result['min_samples_leaf'], max_depth=result['max_depth'],
                                   criterion=result['criterion']))[0])
    print(model.cross_val_model(DecisionTreeClassifier())[0])


labels = db_setup.all_labels()
original_labels = labels
original_labels = list(original_labels)

for i in range(0, len(labels)):
    if labels[i] == "Amazon Echo Dot":
        labels[i] = "Speaker"
    elif labels[i] == "Epson XP 340":
        labels[i] = "Printer"
    elif labels[i] == "Lasko Fan":
        labels[i] = "Fan"
    elif labels[i] == "Mainstays LED Desk Lamp":
        labels[i] = "Lamp"
    elif labels[i] == "Samsung Galaxy Tab A":
        labels[i] = "Tablet"
    elif labels[i] == "Utilitech Electric Space Heater":
        labels[i] = "Heater"
model.load_models()
models = {"DT": model.dt, "KNN": model.knn}

names = []
accuracies = []

for name, clf in models.items():
    results = model.cross_val_model(clf)

    accuracies.append(results[0] * 100)
    names.append(name)

    print(name)
    graph.graph_matrix(results[1], normalize=False,
                       title="Confusion Matrix for " + name + " classifier",
                       labels=labels)


# graph.graph_bar(names, accuracies)

db_setup.init_db()

pd.set_option('display.max_columns', 20)
pd.set_option('expand_frame_repr', True)

db_setup.cursor.execute(db_setup.statement)
df = pd.read_sql(db_setup.statement, con=db_setup.db)

columns = np.array(model.get_columns())
columns = np.insert(columns, 0, 'label')

table_data = pd.DataFrame()
groups = df.groupby(by='label')

for group in groups:

    data = group[1].drop(['id'], axis=1)
    table_data = table_data.append(
        {columns[0]: labels[original_labels.index(group[0])],
         columns[1]: round(data[columns[1]].mean(), 3),
         columns[2]: round(data[columns[2]].mean(), 3),
         columns[3]: round(data[columns[3]].mean(), 3),
         columns[4]: round(data[columns[4]].mean(), 3),
         columns[5]: round(data[columns[5]].mean(), 3),
         columns[6]: round(data[columns[6]].mean(), 3),
         columns[7]: round(data[columns[6]].mean(), 3),
         columns[8]: round(data[columns[8]].mean(), 3)}, ignore_index=True)

    table_data = table_data[columns]


columns = ['Class', 'Mean (amps)', 'Median (amps)', 'SD (amps)', 'Var (amps)', 'IQR (amps)', 'Mode (amps)',
           'Min (amps)', 'Max (amps)']
graph.graph_table(table_data, columns, "Summary of Statistical Features for each Class")

# Prediction Code
# df = pd.DataFrame()
# df = df.append(
#     {columns[0]: 0.151686, columns[1]: 0.18307, columns[2]: 0.0605465, columns[3]: 0.00366587, columns[4]: 0.13076,
#      columns[5]: 0.07846, columns[6]: 0.07846, columns[7]: 0.20922}, ignore_index=True)
#
# df = df[columns]
#
# print(not np.any([0,0,0,0,0,]))
# dt_dict = {"prediction": model.decode(dt.predict(df)[0])}
# knn_dict = {"prediction": model.decode(knn.predict(df)[0])}
#
# prediction = ""
#
# if(dt_dict["prediction"] == knn_dict["prediction"]):
#     prediction = dt_dict["prediction"] or knn_dict["prediction"]
# else:
#     roll = random.randint(1, 100)
#
#     if roll <= 50:
#         prediction = dt_dict["prediction"]
#     elif roll >= 51:
#         prediction = knn_dict["prediction"]
#
# print(prediction)
#
