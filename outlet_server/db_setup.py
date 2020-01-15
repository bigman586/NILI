import numpy as np

import mysql.connector as mysql

db = None
cursor = None

db_name = "outlet_data"
table_name = "%s.data" % db_name
statement = "SELECT * FROM %s" % table_name


# initialize database and cursor
def init_db():
    global db
    global cursor

    db = mysql.connect(
        host="localhost",
        user="root",
        passwd="Computer468",
        database=db_name
    )

    cursor = db.cursor(buffered=True)


# closes database and cursor
def close_db():
    global cursor
    global db

    cursor.close()
    db.close()

    db = None
    cursor = None


def all_labels():

    init_db()
    cursor.execute("SELECT DISTINCT label FROM %s" % table_name)

    columns = np.asarray(cursor.fetchall())
    labels = []

    for name in columns:
        labels.append(name[0])

    labels.sort()

    close_db()
    return labels
