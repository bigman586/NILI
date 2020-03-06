import time

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
    'p': list(range(1, 50)),
    'metric': ["euclidean", "manhattan", "chebyshev", "minkowski"]
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


def random_optimize(classifer, params):
    prepare()

    # run randomized search
    n_iter_search = 100
    random_search = RandomizedSearchCV(classifier, param_distributions=params,
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


# prepare()
# x_train, x_test, y_train, y_test = train_test_split(model.x_train, model.y_train, test_size=0.2)

# # space = hp.choice('classifier', [
# #     {'model': KNeighborsClassifier,
# #      'param': knn_grid_param
# #      },
# #     {'model': DecisionTreeClassifier,
# #      'param': dt_grid_param
# #      }
# # ])


# def objective_func(args):
#     if args['model'] == KNeighborsClassifier:
#         n_neighbors = args['param']['n_neighbors']
#         algorithm = args['param']['algorithm']
#         leaf_size = args['param']['leaf_size']
#         metric = args['param']['metric']
#         clf = KNeighborsClassifier(n_neighbors=n_neighbors,
#                                    algorithm=algorithm,
#                                    leaf_size=leaf_size,
#                                    metric=metric,
#                                    )
#     elif args['model'] == SVC:
#         C = args['param']['C']
#         kernel = args['param']['kernel']
#         degree = args['param']['degree']
#         gamma = args['param']['gamma']
#         clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)

#     clf.fit(x_train, y_train)
#     y_pred_train = clf.predict(x_train)
#     loss = mean_squared_error(y_train, y_pred_train)
#     print("Test Score:", clf.score(x_test, y_test))
#     print("Train Score:", clf.score(x_train, y_train))
#     print("\n=================")
#     return loss


# space = hp.choice('classifier', [
#     {'model': KNeighborsClassifier,
#      'param': {'n_neighbors':
#                    hp.choice('n_neighbors', range(3, 11)),
#                'algorithm': hp.choice('algorithm', ['ball_tree', 'kd_tree']),
#                'leaf_size': hp.choice('leaf_size', range(1, 50)),
#                'metric': hp.choice('metric', ["euclidean", "manhattan",
#                                               "chebyshev", "minkowski"
#                                               ])}
#      },
#     {'model': SVC,
#      'param': {'C': hp.lognormal('C', 0, 1),
#                'kernel': hp.choice('kernel', ['rbf', 'poly', 'rbf', 'sigmoid']),
#                'degree': hp.choice('degree', range(1, 15)),
#                'gamma': hp.uniform('gamma', 0.001, 10000)}
#      }
# ])
# best_classifier = fmin(objective_func, space,
#                        algo=tpe.suggest, max_evals=100)
# print(best_classifier)

# TODO: add legend
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

prepare()


def graph_table():
    labels = change_labels()[1]

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

    print(table_data)
    columns = ['Class', 'Mean (amps)', 'Median (amps)', 'SD (amps)', 'Var (amps)', 'IQR (amps)', 'Mode (amps)',
               'Min (amps)', 'Max (amps)']
    graph.graph_table(table_data, columns,
                      "Summary of Statistical Features for each Class")

# Confusion Matrix Code


def graph_matrix():
    models, labels = change_labels()

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


def change_labels():
    labels = db_setup.all_labels()

    return replace(labels)


def replace(original_labels):
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
    return labels


labels = []
dt_accuracies = []
knn_accuracies = []

for group in groups:
    data = group[1]

    x_train = data.drop(['id', 'label'], axis=1)
    y_train = data['label']

    y_labels = y_train

    y_train = pd.get_dummies(
        y_train, columns=model.categoricals, dummy_na=True)
    y_train.drop(['NaN'], axis=1, errors='ignore')

    model.load_models()

    labels.append(group[0])

    dt_accuracy = model.accuracy_by_class(model.dt, x_train, y_train)
    knn_accuracy = model.accuracy_by_class(model.knn, x_train, y_train)

    dt_accuracies.append(dt_accuracy * 100)
    knn_accuracies.append(knn_accuracy * 100)

    print(group[0])
    print("DT classifier - {}".format(dt_accuracy))
    print("KNN classifier - {}".format(knn_accuracy))

labels = replace(labels)
print(labels)

if (not(dt_accuracies == knn_accuracies == [])):
    graph.graph_group_bar(labels, dt_accuracies, knn_accuracies, "Identification Accuracy per Appliance")

# columns = ['Class', 'Mean (amps)', 'Median (amps)', 'SD (amps)', 'Var (amps)', 'IQR (amps)', 'Mode (amps)',
#            'Min (amps)', 'Max (amps)']
# graph.graph_table(table_data, columns, "Summary of Statistical Features for each Class")

# Prediction Code
"""df = pd.DataFrame()
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

print(prediction) """
