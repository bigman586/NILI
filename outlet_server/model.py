import datetime
import pandas as pd
from sklearn.metrics import accuracy_score

import db_setup
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
import time

pd.set_option('display.max_columns', 20)
pd.set_option('expand_frame_repr', True)

modelNames = ['dt.pkl', 'knn.pkl']


def loadData():
    global df
    global categoricals
    global labels

    db_setup.initDB()

    pd.set_option('display.max_columns', 20)
    pd.set_option('expand_frame_repr', True)

    db_setup.cursor.execute(db_setup.statement)
    df = pd.read_sql(db_setup.statement, con=db_setup.db)

    categoricals = []  # going to one-hot encode categorical variables
    labels = db_setup.allLabels()

    for col, col_type in df.dtypes.items():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic


def prepareData():
    # TODO: Add dataframe prediction
    global x_train
    global y_train

    x_train = df.drop(['id', 'label'], axis=1)
    y_train = df['label']

    # get_dummies effectively creates one-hot encoded variables
    y_train = pd.get_dummies(y_train, columns=categoricals, dummy_na=True)
    y_train.drop(['NaN'], axis=1, errors='ignore')


def decode(prediction):
    result = (np.where(prediction == 1))[0][0]
    return result


def crossVal(model):
    global x_train
    global y_train

    loadData()
    prepareData()

    start = time.time()

    validationSize = 0.20
    seed = 4

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x_train, y_train,
                                                                                    test_size=validationSize,
                                                                                    random_state=seed)

    # testing options and resetting random seed
    seed = 12
    scoring = 'accuracy'

    # checks accuracy of ML method and outputs accuracy percentage dataset n_splits times
    # scores machine learning model on the dataset
    kfold = model_selection.KFold(n_splits=10, random_state=seed,
                                  shuffle=True)  # n_splits - 1 datasets for training, 1 for validation
    cvResults = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    print(('Mean: %s  SD: %f') % (cvResults.mean(), cvResults.std()))
    # print(cvResults)

    # fit training data into decision Tree
    model.fit(X_train, Y_train)
    print("Trained in " + str(time.time() - start) + " seconds")

    # get predictions from machine learning model
    predictions = model.predict(X_validation)

    # check how accurate the model is using the predictions
    accuracy = accuracy_score(Y_validation, predictions, normalize=True)

    labelPrediction = []
    YVal = []

    for i in range(len(predictions)):
        if predictions[i].any():
            labelPrediction.append(labels[decode(predictions[i])])
            # iloc for integer-location based indexing
            YVal.append(
                Y_validation.iloc[i][Y_validation.iloc[i] == 1].index[0])
    # print(YVal)
    # print(labelPrediction)

    print("Accuracy = " + str(accuracy))
    print("")

    return accuracy

    # confusion matrix
    # confusionMat = confusion_matrix(YVal, labelPrediction)
    # print(confusionMat)

    # TODO: add tuple


def train():
    global dt
    global knn

    prepareData()
    loadModels()

    start = time.time()

    dt.fit(x_train, y_train)
    knn.fit(x_train, y_train)

    print("Trained in " + str(time.time() - start) + " seconds")


def predict():
    loadModels()
    train()
    if dt or knn:
        print('Prediction Sent')
    else:
        return


def loadModels():
    global dt
    global knn

    try:
        dt = joblib.load(modelNames[0])
        knn = joblib.load(modelNames[1])
        print('Model loaded')

    except Exception as e:
        print('No model here')
        print(str(e))
        return


def saveModels():
    joblib.dump(dt, modelNames[0])
    joblib.dump(knn, modelNames[1])
    print("Models Saved")

# TODO: Add Cross Validation
