import random
import time

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import (Imputer, LabelEncoder, OneHotEncoder,
                                   StandardScaler)

import db_setup

pd.set_option('display.max_columns', 20)
pd.set_option('expand_frame_repr', True)

model_names = ['dt.joblib', 'knn.joblib']
categoricals = []  # going to one-hot encode categorical variables

dt = None
knn = None


def load_data():
    global df
    global categoricals
    global labels

    db_setup.init_db()

    pd.set_option('display.max_columns', 20)
    pd.set_option('expand_frame_repr', True)

    db_setup.cursor.execute(db_setup.statement)
    df = pd.read_sql(db_setup.statement, con=db_setup.db)

    labels = db_setup.all_labels()

    for col, col_type in df.dtypes.items():
        if col_type == 'O':
            categoricals.append(col)
        else:
            # fill NA's with 0 for ints/floats, too generic
            df[col].fillna(0, inplace=True)


def prepare_data():
    global x_train
    global y_train
    global categoricals
    global df
    global y_labels

    x_train = df.drop(['id', 'label'], axis=1)
    y_labels = df['label']

    # get_dummies effectively creates one-hot encoded variables
    y_train = pd.get_dummies(y_labels, columns=categoricals, dummy_na=True)
    y_train.drop(['NaN'], axis=1, errors='ignore')


def get_columns():
    global categoricals

    load_data()
    prepare_data()

    columns = []
    for col in x_train.columns:
        columns.append(col)

    # print(columns)
    return columns


def cross_val_model(model):
    global x_train
    global y_train
    global labels
    global y_labels

    load_data()
    prepare_data()

    return cross_val(model, x_train, y_train)


def cross_val(model, x_data, y_data):

    validation_size = 0.25
    seed = 3

    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x_data, y_data,
                                                                                    test_size=validation_size,
                                                                                    random_state=seed)
    # print(x_train.shape)
    # print(x_validation.shape)
    # print(y_train.shape)
    # print(y_validation.shape)

    # testing options and resetting random seed
    seed = 3
    scoring = 'accuracy'

    # checks accuracy of ML method and outputs accuracy percentage current_dataset n_splits times
    # scores machine learning model on the current_dataset
    K = 10
    kfold = model_selection.KFold(n_splits=K, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(
        model, x_train, y_train, cv=kfold, scoring=scoring)

    # print(('Mean: %s  SD: %f') % (cv_results.mean(), cv_results.std()))
    # print(cv_results)
    start = time.time()

    # fit training data into classifier
    model.fit(x_train, y_train)
    print("Trained in " + str(time.time() - start) + " seconds")

    # get predictions from machine learning model
    predictions = model.predict(x_validation)
    print(predictions)
    # # print(y_train)

    label_prediction = []
    y_val = []

    for i in range(len(predictions)):
        if predictions[i].any():
            label_prediction.append(decode(predictions[i]))

            # iloc for integer-location based indexing
            y_val.append(y_validation.iloc[i]
                         [y_validation.iloc[i] == 1].index[0])

    # check how accurate the model is using the predictions
    accuracy = accuracy_score(y_val, label_prediction, normalize=True)

    # print(y_val)
    # print(label_prediction)

    # print("Accuracy = " + str(accuracy))
    # print("")

    # confusion matrix
    confusion_mat = confusion_matrix(y_val, label_prediction)
    return (accuracy, confusion_mat, cv_results)


def accuracy_by_class(model, x_data, y_data):
    global x_train
    global y_train

    x_validation = x_data
    y_validation = y_data

    # testing options and resetting random seed
    seed = 3
    scoring = 'accuracy'

    # checks accuracy of ML method and outputs accuracy percentage current_dataset n_splits times
    # scores machine learning model on the current_dataset
    K = 10
    kfold = model_selection.KFold(n_splits=K, random_state=seed,
                                  shuffle=True)
    start = time.time()

    # fit training data into classifier
    model.fit(x_train, y_train)
    print("Trained in " + str(time.time() - start) + " seconds")

    # get predictions from machine learning model
    predictions = model.predict(x_validation)
    # print(predictions)
    # # print(y_train)

    label_prediction = []
    y_val = []

    for i in range(len(predictions)):
        if predictions[i].any():
            label_prediction.append(decode(predictions[i]))

            # iloc for integer-location based indexing
            y_val.append(y_validation.iloc[i]
                         [y_validation.iloc[i] == 1].index[0])

    # print(y_val)
    # print(label_prediction)

    # check how accurate the model is using the predictions
    accuracy = accuracy_score(y_val, label_prediction, normalize=True)

    # print("Accuracy = " + str(accuracy))
    # print("")

    # confusion matrix
    confusion_mat = confusion_matrix(y_val, label_prediction)
    return accuracy


def decode_predictions(predictions):
    label_prediction = []
    y_val = []

    for i in range(len(predictions)):
        if predictions[i].any():
            label_prediction.append(decode(predictions[i]))

            # iloc for integer-location based indexing
            y_val.append(y_validation.iloc[i]
                         [y_validation.iloc[i] == 1].index[0])

    return label_predictions, y_val


def train():
    global dt
    global knn

    load_data()
    prepare_data()
    load_models()

    start = time.time()

    dt.fit(x_train, y_train)
    knn.fit(x_train, y_train)
    
    print("Trained in " + str(time.time() - start) + " seconds")


def decode(prediction):
    db_setup.init_db()

    if (not np.any(prediction)):
        result_text = "Electronic Device"
    else:
        result = (np.where(prediction == 1))[0]
        result = result[0]

        # print(result)
        result_text = db_setup.all_labels()[result]

    db_setup.close_db()
    return result_text


def predict(pred_data):
    global dt
    global knn

    load_models()

    print(pred_data)

    if dt and knn:
        dt_dict = {"prediction": decode(dt.predict(pred_data)[0])}
        knn_dict = {"prediction": decode(knn.predict(pred_data)[0])}

        prediction = ""

        if (dt_dict["prediction"] == knn_dict["prediction"]):
            prediction = dt_dict["prediction"] or knn_dict["prediction"]
        else:
            roll = random.randint(1, 100)

            if roll <= 50:
                prediction = dt_dict["prediction"]
            elif roll >= 51:
                prediction = knn_dict["prediction"]

        print(prediction)
        return prediction
    else:
        return "No prediction"


def load_models():
    global dt
    global knn

    try:
        dt = joblib.load(model_names[0])
        knn = joblib.load(model_names[1])
        print('Models loaded')

    except Exception as e:
        print('No model here')
        print(str(e))
        return


def save_models():
    global dt
    global knn

    if dt or knn:
        joblib.dump(dt, model_names[0])
        joblib.dump(knn, model_names[1])
        print("Models Saved")
    else:
        return "Can't save model"
