# -*- coding: utf-8 -*-
# https://www.tensorflow.org/tutorials/structured_data/feature_columns
# USAGE:
"""
python classify.py batteryvoltage
python classify.py engine_ON
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

def initializeArgs(option):
    args = {}
    args['mainpath'] = '/home/saghar/IG/projects/DrowningWarning-analytics'
    args['training_data'] = os.path.join(args['mainpath'], 'data/merged/capture_246058.txt')
    args['target_col'] = option
    args['trained_model'] = os.path.join(args['mainpath'], 'models/{}.model'.format(args['target_col']))
    args['plotdir'] = os.path.join(args['mainpath'], 'plots/{}'.format(args['target_col']))          
    if option == 'batteryvoltage':
        args['batch_size'] = 16
        args['EPOCHS'] = 300
        args['metrics'] = [
            tf.keras.metrics.CategoricalHinge(name='hinge'),
            tf.keras.metrics.CategoricalCrossentropy(name='crossentropy')
        ]
        args['drop_columns'] = ['timestamp', 'time', 'diff_ms',
                                'temp', 'batt', 'engine_ON', 'speed_knots', 'lat', 'long']
    elif option == 'engine_ON':
        args['batch_size'] = 32
        args['EPOCHS'] = 300
        args['metrics'] = [
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
        ]
        args['drop_columns'] = ['timestamp', 'time', 'diff_ms', 'temp',
                                'batt', 'batteryvoltage', 'speed_knots', 'lat', 'long']
    return args

# A utility method to create a tf.data dataset from a Pandas Dataframe


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(('target'))
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def prepareData(args):
    # read
    dataframe = pd.read_csv(args['training_data'])
    if args['target_col'] == 'batteryvoltage':
        dataframe['target'] = dataframe[args['target_col']].apply(
            lambda x: int(x*10 - 12))  # FIXME
    elif args['target_col'] == 'engine_ON':
        dataframe['target'] = dataframe[args['target_col']] > 0.5

    n_classes = dataframe['target'].nunique()
    print('N_CLASSES:', n_classes)
    dataframe.drop(columns=args['target_col'], inplace=True)
    dataframe.drop(columns=args['drop_columns'], inplace=True)
    # make feature columns
    feature_cols = [col for col in list(dataframe.columns) if col != 'target']
    print('FEATURES: ', feature_cols)
    # scale features
    scaler = MinMaxScaler()
    scaler.fit(dataframe[feature_cols])
    dataframe[feature_cols] = scaler.fit_transform(dataframe[feature_cols])
    # save the scaler to use when testing the model
    joblib.dump(scaler, args['trained_model'].replace('.model', '.scaler.pkl'))
    # make feature layer
    feature_columns = [feature_column.numeric_column(
        col) for col in list(dataframe.columns) if col != 'target']
    #feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    # split train, valid, test sets
    train, test = train_test_split(dataframe, test_size=0.3)
    train, val = train_test_split(train, test_size=0.3)
    # save the test set for evaluating the model later
    test.reset_index(level=[0], drop=True).to_pickle(
        '{}.test.pkl'.format(args['training_data']))
    train_ds = df_to_dataset(train, batch_size=args['batch_size'])
    val_ds = df_to_dataset(val, shuffle=False, batch_size=args['batch_size'])
    test_ds = df_to_dataset(test, shuffle=False, batch_size=args['batch_size'])
    # Using class weights for the loss function (relevant for binary classification of imbalanced data)
    if n_classes == 2:
        neg, pos = np.bincount(train['target'])
        weight_for_false = 1 / neg
        weight_for_true = 1 / pos
        class_weight = {0: weight_for_false, 1: weight_for_true}
    else:
        class_weight = None
    return feature_columns, n_classes, scaler, train_ds, val_ds, test_ds, class_weight


def fitModel(args, feature_columns, n_classes, train_ds, val_ds, class_weight):
    if args['target_col'] == 'batteryvoltage':
        model = tf.keras.Sequential([
            tf.keras.layers.DenseFeatures(feature_columns),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dense(n_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=args['metrics'])

    elif args['target_col'] == 'engine_ON':
        model = tf.keras.Sequential([
            tf.keras.layers.DenseFeatures(feature_columns),          
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=args['metrics'])

    if n_classes == 2:
        history = model.fit(train_ds,
                            validation_data=val_ds,
                            class_weight=class_weight,
                            epochs=args['EPOCHS'])
    else:
        history = model.fit(train_ds,
                            validation_data=val_ds,
                            epochs=args['EPOCHS'])

    return model, history


def evaluateHistory(args, history):
    epochs = range(args['EPOCHS'])

    _ = plt.figure()
    plt.title('Loss')
    plt.plot(epochs, history.history['loss'], color='blue', label='Train')
    plt.plot(epochs, history.history['val_loss'], color='orange', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args['plotdir'], 'loss.png'))

    for metric in [m.name for m in args['metrics']]:
        _ = plt.figure()
        plt.title(metric)
        plt.plot(epochs, history.history[metric], color='blue', label='Train')
        plt.plot(epochs, history.history['val_{}'.format(
            metric)], color='orange', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join(args['plotdir'], '{}.png'.format(metric)))


def main(option):
    args = initializeArgs(option)
    feature_columns, n_classes, scaler, train_ds, val_ds, test_ds, class_weight = prepareData(
        args)
    model, history = fitModel(
        args, feature_columns, n_classes, train_ds, val_ds, class_weight)
    evaluateHistory(args, history)
    model.summary()
    model.save(args['trained_model'])

if __name__ == "__main__":
    main(sys.argv[1])
