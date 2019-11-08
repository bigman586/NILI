import datetime
import pandas as pd
import db_setup
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

pd.set_option('display.max_columns', 20)
pd.set_option('expand_frame_repr', True)

modelNames = ['dt.pkl', 'knn.pkl']


def train():
    global dt
    global knn

    db_setup.initDB()

    # select all data from database
    db_setup.cursor.execute("SELECT * FROM outlet_data.data")
    df = pd.read_sql("SELECT * FROM outlet_data.data", con=db_setup.db)

    db_setup.closeDB()

    X = df.drop(['id', 'label'], axis=1)
    Y = df['label']

    # one-hot encoding
    Y = pd.get_dummies(Y)

    dt = DecisionTreeClassifier()
    knn = KNeighborsClassifier()

    dt.fit(X, Y)
    knn.fit(X, Y)


def predict():
    loadModels()
    if dt or knn:
        print('Prediction Sent')
    else:
        return


def loadModels():
    global dt
    global knn

    dt = joblib.load(modelNames[0])
    knn = joblib.load(modelNames[1])


def saveModels():
    joblib.dump(dt, modelNames[0])
    joblib.dump(knn, modelNames[1])
