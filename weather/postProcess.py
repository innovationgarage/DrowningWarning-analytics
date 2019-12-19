import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def scaleParam(df, oldparam, newparam):
    scaler = MinMaxScaler()
    df[newparam] = scaler.fit_transform(df[oldparam].values.reshape(-1,1))
    return df

def postProcess(filepath):
    df = pd.read_csv(filepath)
    #smooth
    window_length = 51
    polyorder = 3
    df['xsmooth'] = savgol_filter(df['x'], window_length, polyorder)
    df['ysmooth'] = savgol_filter(df['y'], window_length, polyorder) 
    df['bssmooth'] = savgol_filter(df['speed_knots'], window_length, polyorder)
    df['wssmooth'] = savgol_filter(df['wind_speed'], window_length, polyorder) 
    #scale
    df = scaleParam(df, 'bssmooth', 'boatspeed')
    df = scaleParam(df, 'wssmooth', 'windspeed')
    return df

in_filepath ='../data/merged/capture_246058_wind.txt'
boat = postProcess(in_filepath)
plt.figure(figsize=(15,5))
plt.plot(boat.time, boat.boatspeed, lw=2, label='Boat speed')
plt.plot(boat.time, boat.windspeed, lw=3, label='Wind speed')
plt.xlabel('time (unix)')
plt.ylabel('Speed (scaled)')
plt.legend()
plt.savefig('../plots/weather/boat_wind_speed.png')
plt.show()
