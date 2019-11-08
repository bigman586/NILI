import mysql.connector as mysql

db = None
cursor = None


# initialize database
def initDB():
    global db
    global cursor

    db = mysql.connect(
        host="localhost",
        user="root",
        passwd="Computer468",
        database="outlet_data"
    )

    cursor = db.cursor(buffered=True)


def closeDB():
    db.close()
    cursor.close()
