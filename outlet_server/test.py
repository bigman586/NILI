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


db_setup.initDB()

pd.set_option('display.max_columns', 20)
pd.set_option('expand_frame_repr', True)

db_setup.cursor.execute("SELECT * FROM outlet_data.data")
df = pd.read_sql("SELECT * FROM outlet_data.data", con=db_setup.db)

#print(df.columns)

X = df.drop(['id', 'label'], axis=1)
Y = df['label']

categoricals = []  # going to one-hot encode categorical variables

for col, col_type in df.dtypes.items():
    if col_type == 'O':
        categoricals.append(col)
    else:
        df[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

# get_dummies effectively creates one-hot encoded variables
Y = pd.get_dummies(Y, columns=categoricals, dummy_na=True)

knn = KNeighborsClassifier()
start = time.time()

knn.fit(X, Y)


'''dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()

dt.fit(X, Y)
knn.fit(X, Y)'''

print("Trained in " + str(time.time()-start) + " seconds")
