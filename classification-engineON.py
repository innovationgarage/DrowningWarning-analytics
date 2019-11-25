# -*- coding: utf-8 -*-
"""DWClassification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zJMxUlGEai8Tg-CQIF3qEaug4z9ZOE6c
"""
import tensorflow as tf

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

training_data = 'merged/capture_246058.txt'
trained_model = 'engineOn_local.h5'
plotdir = 'plots/model'

#https://www.tensorflow.org/tutorials/structured_data/feature_columns
dataframe = pd.read_csv(training_data)
dataframe['target'] = dataframe['engine_ON'] > 0.5

dataframe.drop(columns = ['timestamp', 'time', 'diff_ms', 'temp', 'batt','engine_ON', 'batteryvoltage', 'speed_knots', 'lat', 'long'], inplace=True)

feature_cols = [col for col in list(dataframe.columns) if col!='target']
print('FEATURES: ', feature_cols)

scaler = MinMaxScaler()
dataframe[feature_cols] = scaler.fit_transform(dataframe[feature_cols])
print(dataframe.head())

train, test = train_test_split(dataframe, test_size=0.3)
train, val = train_test_split(train, test_size=0.3)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

feature_columns = [feature_column.numeric_column(col) for col in list(dataframe.columns) if col!='target']

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# Using class weights for the loss function
neg, pos = np.bincount(train['target'])

weight_for_false = 1 / neg
weight_for_true = 1 / pos

class_weight = {0: weight_for_false, 1: weight_for_true}

print('Weight for class false: {:.2e}'.format(weight_for_false))
print('Weight for class true: {:.2e}'.format(weight_for_true))

metrics = [
           tf.keras.metrics.Precision(name='precision'),
           tf.keras.metrics.Recall(name='recall'),
           tf.keras.metrics.AUC(name='auc'),
           tf.keras.metrics.TruePositives(name='tp'),
           tf.keras.metrics.FalsePositives(name='fp'),
           tf.keras.metrics.TrueNegatives(name='tn'),
           tf.keras.metrics.FalseNegatives(name='fn')
]
#metrics = [ tf.keras.metrics.Accuracy(name='accuracy') ]

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
                metrics = metrics
              )

EPOCHS = 200
history = model.fit(train_ds,
                    validation_data=val_ds,
                    class_weight=class_weight,
                    epochs=EPOCHS)

epochs = range(EPOCHS)

_ = plt.figure()
plt.title('Loss')
plt.plot(epochs, history.history['loss'], color='blue', label='Train')
plt.plot(epochs, history.history['val_loss'], color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(plotdir, 'loss.png'))

_ = plt.figure()
plt.title('False Negative Rate')
plt.plot(epochs, np.array(history.history['fn'])/(np.array(history.history['fn'])+np.array(history.history['tn'])), color='blue', label='Train')
plt.plot(epochs, np.array(history.history['val_fn'])/(np.array(history.history['val_fn'])+np.array(history.history['val_tn'])), color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('False Negative Rate')
plt.legend()
plt.savefig(os.path.join(plotdir, 'fnr.png'))

_ = plt.figure()
plt.title('True Positive Rate')
plt.plot(epochs, np.array(history.history['tp'])/(np.array(history.history['tp'])+np.array(history.history['fn'])), color='blue', label='Train')
plt.plot(epochs, np.array(history.history['val_tp'])/(np.array(history.history['val_tp'])+np.array(history.history['val_fn'])), color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig(os.path.join(plotdir, 'tpr.png'))

#Evaluate the baseline model
results = model.evaluate(test_ds)
for name, value in zip(model.metrics_names, results):
  print(name, ': ', value)

model.save(trained_model)

#Examine the confusion matrix
predicted_labels = model.predict(test_ds)
cm = confusion_matrix(test['target'], np.round(predicted_labels))

plt.matshow(cm, alpha=0)
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

for (i, j), z in np.ndenumerate(cm):
    plt.text(j, i, str(z), ha='center', va='center')

plt.savefig(os.path.join(plotdir, 'confusion.png'))    


print('True Negatives: ', cm[0][0])
print('False Positives: ', cm[0][1])
print('False Negatives: ', cm[1][0])
print('True Positives: ', cm[1][1])

test['proba'] = predicted_labels
lims = np.linspace(0.1,0.9, 9)
ts = np.zeros_like(lims)
fs = np.zeros_like(lims)
for i, lim in enumerate(lims):
  test['predicted'] = test['proba'] > lim
  ts[i] = test[test['target']==test['predicted']].shape[0]
  fs[i] = test[test['target']!=test['predicted']].shape[0]

print(test[test['target']==test['predicted']].shape[0])
print(test[test['target']!=test['predicted']].shape[0])

plt.figure(figsize=(22,5))
plt.plot(lims, fs, 'r', lw=3, label='false')
plt.plot(lims, ts, 'g', lw=3, label='true')
plt.legend()
plt.savefig(os.path.join(plotdir, 'predict.png'))


