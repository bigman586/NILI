import mysql.connector as mysql
import numpy as np

import config

db = None
cursor = None

db_name = config.DB_NAME
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


def all_labels():
    """
    :return: all unique labels arranged alphabetically
    """

    init_db()
    cursor.execute("SELECT DISTINCT label FROM %s" % table_name)

    columns = np.asarray(cursor.fetchall())
    labels = []

    for name in columns:
        labels.append(name[0])

    # arrange labels alphabetically
    labels.sort()

    close_db()
    return labels
