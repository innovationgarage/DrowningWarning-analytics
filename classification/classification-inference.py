import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def initializeArgs(option):
    args = {}
    args['test_data'] = '../data/merged/capture_246058.txt.test.pkl'
    args['trained_model'] = 'testmodel.h5'
    args['plotdir'] = '../plots/BV'
    args['scaler'] = args['trained_model'].replace('.h5', '.scaler.pkl')
    return args

def infer(args):
    #load the model
    model = tf.keras.models.load_model('aaa.model')
    return model
    # #load the test data
    # test = pd.read_pickle(args['test_data'])
    # test_labels = test.pop(('target'))
    # test_features = test
    # pred_labels = model.predict([test_features])
    # return model, test_features, test_labels, pred_labels

def main(option):
    args = initializeArgs(option)
    #model, test_features, test_labels, pred_labels = infer(args)
    model = infer(args)
    model.summary()

if __name__=="__main__":
  main(sys.argv[1])
