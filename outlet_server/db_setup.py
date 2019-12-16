import numpy as np

import mysql.connector as mysql

db = None
cursor = None

DBName = "outlet_data"
tableName = "%s.data" % DBName
statement = "SELECT * FROM %s" % tableName

#initialize database and cursor
def initDB():
    global db
    global cursor

    db = mysql.connect(
        host="localhost",
        user="root",
        passwd="Computer468",
        database=DBName
    )

    cursor = db.cursor(buffered=True)

#closes database and cursor
def closeDB():
    db.close()
    cursor.close()

def allLabels():
    initDB()
    cursor.execute("SELECT DISTINCT label FROM %s" % tableName)

    columns = np.asarray(cursor.fetchall())
    labels = []

    for name in columns:
        labels.append(name[0])

    labels.sort()

    closeDB()
    return labels
