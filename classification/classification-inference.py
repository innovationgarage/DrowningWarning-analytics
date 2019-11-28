import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def initializeArgs(option):
    args = {}
    args['mainpath'] = '/home/saghar/IG/projects/DrowningWarning-analytics'
    args['target_col'] = option
    args['test_data'] = os.path.join(args['mainpath'], 'data/merged/capture_246058.txt.test.pkl')
    args['trained_model'] = os.path.join(args['mainpath'], 'models/{}.model'.format(args['target_col']))
    args['plotdir'] = os.path.join(args['mainpath'], 'plots/{}'.format(args['target_col']))
    args['scaler'] = args['trained_model'].replace('.model', '.scaler.pkl')
    return args

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(('target'))
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

def infer(args):
    #load the model
    model = tf.keras.models.load_model(args['trained_model'])
    #load the test data
    test = pd.read_pickle(args['test_data'])
    test_ds = df_to_dataset(test, shuffle=False)
    pred_labels = model.predict(test_ds)
    if args['target_col'] == 'engine_ON':
        test['engineON_proba'] = pred_labels
        test['engineON_pred'] = test['engineON_proba'] > 0.5
    elif args['target_col'] == 'batteryvoltage':
        for i in range(pred_labels.shape[1]):
            test['class_{}'.format(i)] = pred_labels[:,i]
        test['bv_pred'] = test[['class_0', 'class_1', 'class_2', 'class_3']].idxmax(axis=1)
        test['bv_pred'] = test['bv_pred'].apply(lambda x: int(x.replace("class_", "")))

    return model, test

def main(option):
    args = initializeArgs(option)
    model, test = infer(args)
    print(test.head())

if __name__=="__main__":
  main(sys.argv[1])
