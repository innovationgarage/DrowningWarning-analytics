import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import sklearn.decomposition
import scipy.signal

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib.pyplot as plt
import numpy as np
import datetime


def load(name="246058"):
    global merged, data
    merged=pd.read_csv("../data/merged/capture_%s.txt" % name, delimiter=",", parse_dates=['timestamp'])
    data=pd.read_csv("../data/clean/capture.capture_%s.txt" % name, delimiter=",", parse_dates=['timestamp'])
    data.set_index('timestamp', inplace=True)
    data = data.resample("0.1S").mean()
    data = data.interpolate()

def plot_raw_data():
    plt.figure(figsize=(20, 10))
    plt.plot(data["ax"], label="Acceleration X")
    plt.plot(data["ay"], label="Acceleration Y")
    plt.plot(data["az"], label="Acceleration Z")
    plt.legend()
    plt.title("Raw sensor data for acceleration")
    plt.xlabel("Time")
    plt.show()


def plot_frequency_vs_lat():
    starttime = data.index.min()
    endtime = data.index.max()

    p = sklearn.decomposition.PCA(1)
    d = p.fit_transform(data[["ax", "ay", "az"]].values)

    M=400
    f, t, Sxx = scipy.signal.spectrogram(
        d[:,0],
        window=scipy.signal.windows.kaiser(M, 5),
        nperseg=M, noverlap=M/2, fs=10)

    t = np.array(starttime, dtype="datetime64") + t.astype("timedelta64[s]")

    w = np.where((t > datetime.datetime(2019, 10, 11, 11, 30)) & (t < datetime.datetime(2019, 10, 11, 16, 30)))[0]
    tt = t[w]
    ss = Sxx[:,w]
    dd = d[::M//2,0][:-2]
    dd = dd[w]
    dd = dd - dd.min()
    dd = dd / dd.max() * f.max()

    plt.figure(figsize=(20, 10))
    plt.pcolormesh(tt, f, ss, vmin=ss.mean() - ss.std(), vmax=ss.mean() + ss.std())

    lats = merged.lat.values - merged.lat.values.min()
    lats = f.max() * lats / lats.max()
    plt.plot(merged.timestamp.values, lats, color="blue", label="Latitude")

    plt.title("Rocking motion vs latitude part of position of boat on map")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Timestamp')
    plt.legend()
    plt.show()


def plot_frequency_vs_speed():
    starttime = data.index.min()
    endtime = data.index.max()

    p = sklearn.decomposition.PCA(1)
    d = p.fit_transform(data[["ax", "ay", "az"]].values)

    M=400
    f, t, Sxx = scipy.signal.spectrogram(
        d[:,0],
        window=scipy.signal.windows.kaiser(M, 5),
        nperseg=M, noverlap=M/2, fs=10)

    t = np.array(starttime, dtype="datetime64") + t.astype("timedelta64[s]")

    pdSxx = pd.DataFrame(Sxx.transpose(), index=t, columns=f)
    pdSxx["timestamp"] = pdSxx.index

    speeds = merged[["timestamp", "speed_knots"]]
    speeds = speeds.set_index("timestamp")
    speeds = speeds.tz_localize(None)

    speedgraph = pd.merge(pdSxx, speeds, how='left', on='timestamp', sort=True)
    speedgraph = speedgraph.set_index("timestamp")
    speedgraph = speedgraph.interpolate()

    speedgraph = speedgraph.dropna()
    speedgraph = speedgraph.set_index("speed_knots").sort_index()

    plt.figure(figsize=(20, 20))
    v = speedgraph.values
    plt.pcolormesh(speedgraph.columns, speedgraph.index, v, vmin=v.mean()-v.std(), vmax=v.mean()+v.std())
    plt.title("Rocking motion vs speed of boat")
    plt.ylabel('Speed')
    plt.xlabel('Frequency [Hz]')
    plt.show()
