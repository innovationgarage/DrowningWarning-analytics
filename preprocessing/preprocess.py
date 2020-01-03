import pytz
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
from geopy.distance import distance
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

"""
UDAGE:
python clean_data.py --ti telespor.csv --to telespor_clean.csv --ci capture.txt --co capture_clean.csv --allout all_data.csv --starttime '2019-10-11 10:17:00.0000'
"""

#Get and clean data that comes from telespor API eport
def readTelespor(infile):
    df = pd.read_csv(infile, delimiter=";")
    df.columns = ['ID', 'timestamp', 'seqnr', 'persnr', 'telespornr', 'sender',
                  'receiver', 'position, valid', 'long', 'lat', 'position_time',
                  'Debug 1', 'Debug 2', 'batteryvoltage', 'temperature', 'Alarm status',
                  'Alarm time ', 'UHF TX effect', 'dispersiontime', 'fieldstrength UHF',
                  'fieldstrength VHF', 'fieldstrength GSM', 'Debug', 'Firmware']
    to_drop = ['UHF TX effect', 'sender', 'receiver', 'dispersiontime',
               'telespornr', 'fieldstrength UHF', 'Alarm time ',
               'fieldstrength VHF', 'Alarm status',
               'Firmware', 'Debug', 'position, valid',
               'Debug 2', 'fieldstrength GSM', 'Debug 1',
               'seqnr', 'persnr', 'timestamp']
    df.drop(columns=to_drop, inplace=True)
    df['position_time'] = pd.to_datetime(df['position_time'], utc=True)
    df.sort_values(by='position_time', inplace=True)
    df.rename(columns={'position_time':'timestamp'}, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def measureEngineTime(df):
    df['t-1'] = df['timestamp'].shift(1)
    df['dt'] = df['timestamp'] - df['t-1']
    df['engine_ON'] = df['batteryvoltage']==df.batteryvoltage.max()
    summary = df.groupby('engine_ON')['dt'].sum().reset_index(level=[0])
    return df, summary

def calculateSpeed(df):
    df['t-1'] = df['timestamp'].shift(1)
    df['dt'] = df['timestamp'] - df['t-1']
    df['dt(s)'] = pd.to_numeric(df['dt'])*1e-9
    df['lat-1'] = df['lat'].shift(1)
    df['long-1'] = df['long'].shift(1)
    df.loc[1:,'distance(m)'] = df.loc[1:,['lat', 'long', 'lat-1', 'long-1']].apply(lambda x: distance((x[0], x[1]), (x[2], x[3])).m, axis=1)
    df['speed(m/s)'] = df['distance(m)'] / df['dt(s)']
    df['speed_knots'] = df['speed(m/s)'] * 1.94384449
    df['speed_knots'].fillna(-1, inplace=True)
    df.drop(columns=['t-1', 'dt', 'dt(s)', 'lat-1', 'long-1', 'distance(m)', 'speed(m/s)'], inplace=True)
    return df

def cleanTelespor(infile, outfile, drop_cols):
    df = readTelespor(infile)
    df, summary = measureEngineTime(df)
    print(summary)
    df = calculateSpeed(df)
    df.drop(columns=drop_cols, inplace=True)
    df.to_csv(outfile, index=False)
    return df, df.timestamp.min(), df.timestamp.max()

#Get and clean data captured in the box
def scaleColumns(df, cols):
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def smoothColumns(df, cols):
    savgol_cols = ['{}_savgol'.format(col) for col in cols]
    for i, col in enumerate(cols):
        df[savgol_cols[i]] = savgol_filter(df[col], 101 ,3)
    return df

def cleanCapture(infile, outfile, starttime):
    df = pd.read_csv(infile)
    cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    a_sensitivity = 16384.
    g_sensitivity = 131.
    for col in ['ax', 'ay', 'az']:
        df[col] = df[col]/a_sensitivity #g force
    for col in ['gx', 'gy', 'gz']:
        df[col] = df[col]/g_sensitivity #degree per second

    df = df.sort_values(by='time').reset_index(drop=True) #ms
    start = df['time'].values[0]
    df['dt'] = df['time'] - start 
    df['dt'] = pd.to_timedelta(df['dt'], unit='ms')
    df['timestamp'] = pd.to_datetime(starttime) + df['dt']
    df['timestamp'] = df['timestamp'].dt.tz_localize("Europe/Oslo")
    df['timestamp'] = df['timestamp'].dt.tz_convert("UTC")
    df.to_csv(outfile, index=False)
    return df, df.timestamp.min(), df.timestamp.max()

def resampleAll(capture, telespor, sample_rate, signal_start, signal_end):
    signal_start = pd.to_datetime(signal_start, utc=True)
    signal_end = pd.to_datetime(signal_end, utc=True) 
    
    capture.set_index('timestamp', inplace=True)
    capture = capture.resample(sample_rate).mean().interpolate()
    
    telespor.set_index('timestamp', inplace=True)
    telespor = telespor.resample(sample_rate).mean().interpolate()
    
    telespor.reset_index(level=[0], inplace=True)
    capture.reset_index(level=[0], inplace=True)
    telespor = telespor[telespor.timestamp>=signal_start]
    telespor = telespor[telespor.timestamp<=signal_end]
    telespor.reset_index(level=[0], inplace=True, drop=True)
    capture = capture[capture.timestamp>=signal_start]
    capture = capture[capture.timestamp<=signal_end]
    capture.reset_index(level=[0], inplace=True, drop=True)
    
    return capture, telespor

def main(args):
    interp_cols = ['lat', 'long',
                   'batteryvoltage', 'engine_ON', 'speed_knots',
                   'ax', 'ay', 'az',
                   'gx', 'gy', 'gz']
    drop_cols = ['ID', 'temperature']

    capture, capture_start, capture_end = cleanCapture(args.ci, args.co, args.starttime)
    telespor, telespor_start, telespor_end = cleanTelespor(args.ti, args.to, drop_cols)

    capture, telespor = resampleAll(capture, telespor, args.samplerate, args.signalstart, args.signalend)
    merged = pd.merge(capture, telespor, how='left', on='timestamp', sort=True)

    for col in ['lat', 'long', 'batteryvoltage']:
        merged[col] = merged[col].fillna(method='ffill')
        merged[col] = merged[col].fillna(method='bfill') 
        merged['engine_ON'] = merged['batteryvoltage']==merged.batteryvoltage.max()

    merged.to_csv(args.allout, index=False)        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--ti", default="telespor.csv", help="Raw CSV from Telespor API export")
    parser.add_argument("--to", default = "telespor_clean.csv", help="CSV file containing clean telespor data")
    parser.add_argument("--ci", default="capture.csv", help="Raw CSV from the box capture")
    parser.add_argument("--co", default = "capture_clean.csv", help="CSV file containing clean box capture data")
    parser.add_argument("--allout", default = "all_data.csv", help="CSV file containing the merged data")
    parser.add_argument("--starttime", default = '2019-10-11 10:17:00', help="the start timestamp")
    parser.add_argument("--signalstart", default = "2019-10-11 11:00:00+00:00", help="when the moving signal starts")
    parser.add_argument("--signalend", default = "2019-10-11 17:00:00+00:00", help="when the moving signal end")
    parser.add_argument("--samplerate", default = "10S")
    
    args = parser.parse_args()
    main(args)
    
#######################################TEST###########################################
def test():
    args = {}
    args['ti'] = "data/first/telespor.csv"
    args["to"] =  "data/first/telespor_clean.csv"
    args["ci"] =  "data/first/capture_246058.txt"
    args["co"] =  "data/first/capture_clean.csv"
    args["allout"] =  "all_merged_1_1.csv"
    args["starttime"] = '2019-10-11 10:17:00.000'
    args["signalstart"] = "2019-10-11 11:00:00+00:00"
    args["signalend"] = "2019-10-11 17:00:00+00:00"
    args["samplerate"] = "10S"
    main(args)
