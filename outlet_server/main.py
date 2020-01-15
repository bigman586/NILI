from flask import Flask, request, jsonify, json, render_template
import db_setup
import model
import numpy as np
import pandas as pd
from scipy import stats

app = Flask(__name__)
db_setup.init_db()

sample_label = ""
state = False
current_mean = 0

sampling = []
columns = model.get_columns()
test_data = pd.DataFrame(columns=columns)

table_name = db_setup.table_name
statement = db_setup.statement


@app.route("/")
def home():
    """
    root directory of  server
    """
    return render_template('index.html', current=current_mean, label=sample_label, status=state)


@app.route('/getMean')
def getMean():
    global current_mean
    return str(current_mean)


@app.route('/getLabel')
def getLabel():
    global sample_label
    return str(sample_label)


@app.route('/getStatus')
def getStatus():
    """
    :return: status of device connected to outlet
    """
    global state

    if state:
        return 'on'
    elif not state:
        return 'off'


@app.route('/postCommand', methods=['POST'])
def postCommmand():
    """
    changes status of outlet based on info. from app
    """
    global state

    command_json = request.get_json()
    command_string = str(command_json["command"])

    if (len(command_string) == 0):
        return jsonify({"error': 'invalid input"})

    if command_string == 'on':
        state = True
    if command_string == 'off':
        state = False

    print(str(state))
    return str(state)


def process_data(current):
    """
    inserts data to table

    :param current: float of current value
    :return:
    """
    global sampling

    interval = 10

    if (len(sampling) < interval):
        sampling.append(current)
        # print(current)
    else:
        print(sampling)
        sampling = np.array(sampling)
        sampling = sampling.astype(float)

        process_sample()
        return ("Saved Sampling")


@app.route('/getPrediction')
def getPrediction():
    """

    :return: prediction in json format
    """
    global test_data

    if (not test_data.empty):
        pred_data = pd.DataFrame()
        pred_data = pred_data.append(
            {columns[0]: test_data[columns[0]].mean(), columns[1]: test_data[columns[1]].mean(),
             columns[2]: test_data[columns[2]].mean(),
             columns[3]: test_data[columns[3]].mean(), columns[4]: test_data[columns[4]].mean(),
             columns[5]: test_data[columns[5]].mean(),
             columns[6]: test_data[columns[6]].mean(), columns[7]: test_data[columns[7]].mean()}, ignore_index=True)

        pred_data = pred_data[columns]
        prediction = model.predict(pred_data)

        print(prediction)
    else:
        prediction = "No data available"

    return jsonify(prediction=prediction)


# POST REQUEST
@app.route('/postData', methods=['POST'])
def getData():
    """
    retrieves data from server
    """
    global current_mean

    # converts byte literal to string and processes it
    current = request.get_data().decode("utf-8")
    if current:
        process_data(current)
        current_mean = current

    return ("Data Posted")


# GET REQUEST
@app.route("/getAllLabels")
def getAllLabels():
    """
    :return: all unique labels arranged alphabetically
    """
    all_labels = db_setup.all_labels()
    all_labels = json.dumps(all_labels)

    print(all_labels)

    return all_labels


# POST REQUEST
@app.route("/postLabel", methods=["POST"])
def postLabel():
    """
    gets the label the user inputted as a json value
    """
    global sample_label

    label_json = request.get_json()
    label_string = str(label_json["label"])

    if (len(label_string) == 0):
        return jsonify({"error': 'invalid input"})

    getAllLabels()

    if (sample_label != label_string):
        sample_label = label_string

    print("Label Received: [" + label_string + "]")
    return (label_string + "is currently plugged in")


def get_features():
    mean = np.mean(sampling)
    median = np.median(sampling)
    sd = np.std(sampling)
    variance = np.var(sampling)
    iqr = np.subtract(*np.percentile(sampling, [75, 25]))
    mode = stats.mode(sampling)[0]
    min = np.min(sampling)
    max = np.max(sampling)

    features = dict()
    features[columns[0]] = float(mean)
    features[columns[1]] = float(median)
    features[columns[2]] = float(sd)
    features[columns[3]] = float(variance)
    features[columns[4]] = float(iqr)
    features[columns[5]] = float(mode)
    features[columns[6]] = float(min)
    features[columns[7]] = float(max)

    return features


def process_sample():
    global test_data
    global sampling

    features = get_features()

    if sample_label != "":
        insert_data(features)

    test_data = test_data.append(
        {'mean': features['mean'], 'median': features['median'], 'sd': features['sd'], 'variance': features['variance'],
         'iqr': features['iqr'], 'mode': features['mode'], 'min': features['min'], 'max': features['max']},
        ignore_index=True)

    print(test_data)
    sampling = []


def insert_data(features):
    """
    inserts data into MySQL table
    """
    global test_data
    db_setup.init_db()

    request = "INSERT INTO {0} (label, mean, median, sd, variance, iqr, mode, min, max) " \
              "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)".format(db_setup.table_name)
    values = (
        sample_label, features[columns[0]], features[columns[1]], features[columns[2]], features[columns[3]],
        features[columns[4]],
        features[columns[5]], features[columns[6]], features[columns[7]])

    print(sample_label)

    db_setup.cursor.execute(request, values)

    db_setup.db.commit()
    db_setup.close_db()

    model.train()
    test_data = pd.DataFrame(columns=model.get_columns())


if __name__ == '__main__':
    getAllLabels()
    model.load_models()

    model.load_data()
    model.prepare_data()

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

    # model.save_models()
    print("Server Closed")
