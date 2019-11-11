from flask import Flask, request, jsonify, json, render_template
import db_setup

db_setup.initDB()
import numpy as np
from scipy import stats

app = Flask(__name__)
# socketio = SocketIO(app)

currentLabel = ""
switchOn = False
minCurrent = 10 ** 5
currentMean = 0

tableName = db_setup.tableName
statement = db_setup.statement

@app.route("/")
def home():
    """
    root directory of  server
    """
    return render_template('index.html', current=currentMean, label=currentLabel, status=switchOn)


@app.route('/getMean')
def getMean():
    global currentMean
    return str(currentMean)


@app.route('/getLabel')
def getLabel():
    global currentLabel
    return str(currentLabel)


@app.route('/getStatus')
def getStatus():
    """
    :return: status of device connected to outlet
    """
    global switchOn

    if switchOn:
        return 'on'
    elif not switchOn:
        return 'off'


@app.route('/postCommand', methods=['POST'])
def postCommmand():
    """
    changes status of outlet based on info. from app
    """
    global switchOn

    commandJSON = request.get_json()
    commandString = str(commandJSON["command"])

    if (len(commandString) == 0):
        return jsonify({"error': 'invalid input"})

    if commandString == 'on':
        switchOn = True
    if commandString == 'off':
        switchOn = False

    print(str(switchOn))
    return str(switchOn)


# POST REQUEST
@app.route('/postData', methods=['POST'])
def getData():
    """
    retrieves data from server
    """
    global minCurrent
    global switchOn
    global currentMean

    sampling = []
    # oconverts byte literal to string
    current = request.get_data().decode("utf-8")

    interval = 10

    while (len(sampling) <= interval):
        sampling.append(current)

    if sampling:
        # converts string array to float array
        sampling = np.array(sampling)
        sampling = sampling.astype(float)

        currentMean = np.mean(sampling)

        if currentLabel != "":
            insertData(sampling)

        for current in sampling:
            if current < minCurrent:
                minCurrent = current
                # switchOn = not switchOn

        print(sampling)
    else:
        return ("null")

    # resets sampling
    sampling = []

    # TODO add reference to train function
    return ("Saved Sampling")


'''return jsonify({"prediction": list(map(int, prediction))})'''


# GET REQUEST
@app.route("/getAllLabels")
def getAllLabels():
    """
    :return: all unique labels as an array
    """
    db_setup.initDB()
    db_setup.cursor.execute(statement)

    datasets = db_setup.cursor.fetchall()
    allLabels = []

    for dataset in datasets:
        allLabels.append(dataset[1])

    unique = []

    for label in allLabels:
        if label not in unique:
            unique.append(label)

    allLabels = unique
    # allLabels.sort()

    allLabels = json.dumps(allLabels)

    print(allLabels)
    db_setup.closeDB()

    return allLabels


# POST REQUEST
@app.route("/postLabel", methods=["POST"])
def postLabel():
    """
    gets the label the user inputted as a json value
    """
    global currentLabel

    labelJSON = request.get_json()
    labelString = str(labelJSON["label"])

    if (len(labelString) == 0):
        return jsonify({"error': 'invalid input"})

    getAllLabels()

    if (labelString != currentLabel):
        currentLabel = labelString

    print("Label Received: [" + labelString + "]")
    return (labelString + "is currently plugged in")


def insertData(sampling):
    """
    inserts data into MySQL table
    """
    global currentMean
    db_setup.initDB()

    median = np.median(sampling)
    sd = np.std(sampling)
    variance = np.var(sampling)
    iqr = np.subtract(*np.percentile(sampling, [75, 25]))
    mode = stats.mode(sampling)[0]
    min = np.min(sampling)
    max = np.max(sampling)

    request = "INSERT INTO {0} (label, mean, median, sd, variance, iqr, mode, min, max) " \
              "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)".format(db_setup.tableName)
    values = (
    currentLabel, float(currentMean), float(median), float(sd), float(variance), float(iqr), float(mode), float(min),
    float(max))

    print(currentLabel)
    db_setup.cursor.execute(request, values)

    db_setup.db.commit()
    db_setup.closeDB()


if __name__ == '__main__':
    # socketio.run(app, host='0.0.0.0', port=5000)
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("Server Closed")
