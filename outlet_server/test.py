from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import db_setup
import pandas as pd
from sklearn import preprocessing
import main


def getFeatures():
    features = []

    db_setup.initDB()
    db_setup.cursor.execute("SHOW columns FROM outlet_data.data")
    for column in db_setup.cursor.fetchall():
        if (column[0] != 'id'):
            features.append(column[0])

    db_setup.closeDB()
    return features

db_setup.initDB()

pd.set_option('display.max_columns', 20)
pd.set_option('expand_frame_repr', True)

db_setup.cursor.execute("SELECT * FROM outlet_data.data")
df = pd.read_sql("SELECT * FROM outlet_data.data", con=db_setup.db)

print(df.columns)

XFeatures = getFeatures()[1:]
YFeatures = getFeatures()[0:1]

print(XFeatures)

X = df.drop(['id', 'label'], axis=1)
Y = df['label']

numeric_features = X.columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = Y.columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', DecisionTreeClassifier())])

X = df.drop(['id', 'label'], axis=1)
Y = df['label']

clf.fit(X, Y)


'''dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()

dt.fit(X, Y)
knn.fit(X, Y)'''
