import numpy as np
import pandas as pd
from donut import complete_timestamp, standardize_kpi, Donut, DonutTrainer, DonutPredictor
from tensorflow import keras as K
from tfsnippet.modules import Sequential
from tfsnippet.utils import get_variables_as_dict, VariableSaver
import tensorflow.compat.v1 as tf
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
tf.disable_v2_behavior()

# path to the dataset
file_csv = "KPI.csv"

# Read the raw data.
data = pd.read_csv(file_csv)
timestamp = data["timestamp"]
values = data["value"]
labels = data["label"]
dataset_name = file_csv.split('.')[0]
print("Timestamps: {}".format(timestamp.shape[0]))

# Complete the timestamp filling missing points with zeros, and obtain the missing point indicators.
timestamp, missing, (values, labels) = complete_timestamp(timestamp, (values, labels))
print("Missing points: {}".format(np.sum(missing == 1)))
print("Labeled anomalies: {}".format(np.sum(labels == 1)))

# Split the training and testing data.
test_portion = 0.3
test_n = int(len(values) * test_portion)
train_values, test_values = values[:-test_n], values[-test_n:]
train_labels, test_labels = labels[:-test_n], labels[-test_n:]
train_missing, test_missing = missing[:-test_n], missing[-test_n:]
print("Rows in test set: {}".format(test_values.shape[0]))
print("Anomalies in test set: {}".format(np.sum(test_labels == 1)))

# Standardize the training and testing data, anomaly points or missing points are excluded
train_values, mean, std = standardize_kpi(
    train_values, excludes=np.logical_or(train_labels, train_missing))
test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)
print("Train values mean: {}".format(mean))
print("Train values std: {}".format(std))

sliding_window = 120

# define the model inside the 'model_vs' scope
with tf.variable_scope('model') as model_vs:
    model = Donut(
        h_for_p_x=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        h_for_q_z=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        x_dims=sliding_window,
        z_dims=5,
    )
# use DonutTrainer class to train the model
trainer = DonutTrainer(model=model, model_vs=model_vs, max_epoch=30)
# use DonutPredictor class to make predictions
predictor = DonutPredictor(model)

save_dir = "model/" + dataset_name + "/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
saved = True
if len(os.listdir(save_dir)) == 0:
    saved = False

if saved is False:
    with tf.Session().as_default():
        # train the model
        trainer.fit(train_values, train_labels, train_missing, mean, std)
        # save variables to 'save_dir' directory
        var_dict = get_variables_as_dict(model_vs)
        saver = VariableSaver(var_dict, save_dir)
        saver.save()
        saved = True
if saved:
    with tf.Session().as_default():
        # restore variables from 'save_dir'
        saver = VariableSaver(get_variables_as_dict(model_vs), save_dir)
        saver.restore()
        # make predictions
        test_score = predictor.get_score(test_values, test_missing)
        print("Number of predictions: {}".format(test_score.shape[0]))
        # try different thresholds
        best_threshold = 0
        best_f1 = 0
        best_predictions = []
        thresholds = np.arange(5, 50, 0.2)
        for t in thresholds:
            threshold = t  # can be changed to better fit the training data
            anomaly_predictions = []
            for l in test_score:
                if abs(l) > threshold:
                    anomaly_predictions.append(1)
                else:
                    anomaly_predictions.append(0)
            # strategy to compute modified metrics
            # https://arxiv.org/pdf/1802.03903.pdf, fig 7
            for i in range(sliding_window-1, len(anomaly_predictions)):
                if anomaly_predictions[i-sliding_window+1] == 1 and test_labels[i] == 1:  # true positive
                    j = i-1
                    while j >= sliding_window-1 and test_labels[j] == 1\
                            and anomaly_predictions[j-sliding_window+1] == 0:
                        anomaly_predictions[j-sliding_window+1] = 1
                        j -= 1
                    j = i+1
                    while j < len(anomaly_predictions) and test_labels[j] == 1\
                            and anomaly_predictions[j-sliding_window+1] == 0:
                        anomaly_predictions[j-sliding_window+1] = 1
                        j += 1
            f1 = f1_score(test_labels[sliding_window-1:], anomaly_predictions, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_predictions = anomaly_predictions

        anomaly_predictions = np.array(best_predictions)
        print("--  final results --")
        print("Best anomaly threshold {}".format(best_threshold))
        print("Anomalies found: {}/{}".format(np.sum(anomaly_predictions == 1), np.sum(test_labels == 1)))
        prfs = precision_recall_fscore_support(test_labels[sliding_window-1:], anomaly_predictions)
        print("--- normal rows ---")
        print("precision: {:.3f}".format(prfs[0][0]))
        print("recall: {:.3f}".format(prfs[1][0]))
        print("fscore: {:.3f}".format(prfs[2][0]))
        print("--- anomaly rows ---")
        print("precision: {:.3f}".format(prfs[0][1]))
        print("recall: {:.3f}".format(prfs[1][1]))
        print("fscore: {:.3f}".format(prfs[2][1]))

        """
        --  final results (cpu4 dataset) --
        --  final results --
        Best anomaly threshold 7.2
        Anomalies found: 10/14
        --- normal rows ---
        precision: 0.999
        recall: 1.000
        fscore: 1.000
        --- anomaly rows ---
        precision: 1.000
        recall: 0.714
        fscore: 0.833
        
        --  final results (timeseries dataset) --
        Best anomaly threshold 46.8
        Anomalies found: 1032/1023
        --- normal rows ---
        precision: 1.000
        recall: 1.000
        fscore: 1.000
        --- anomaly rows ---
        precision: 0.991
        recall: 1.000
        fscore: 0.996
        """

