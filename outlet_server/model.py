import datetime
import pandas as pd
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


def train():
    global dt
    global knn

    db_setup.initDB()

    # select all data from database
    db_setup.cursor.execute(db_setup.statement)
    df = pd.read_sql(db_setup.statement, con=db_setup.db)

    db_setup.closeDB()

    X = df.drop(['id', 'label'], axis=1)
    Y = df['label']

    # categorical variables
    categoricals = []

    for col, col_type in df.dtypes.items():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

    # one-hot encoding
    Y = pd.get_dummies(Y, columns=categoricals, dummy_na=True)

    knn = KNeighborsClassifier()

    start = time.time()

    dt.fit(X, Y)
    knn.fit(X, Y)

    print("Trained in " + str(time.time() - start) + " seconds")


def predict():
    loadModels()
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
