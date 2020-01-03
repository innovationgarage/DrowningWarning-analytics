# DrowningWarning-analytics
analyzing data collected from the box (containing accelerometer + gyro sesonrs)

## PreProcessing
The bach script __preprocessAll.sh__ calls the python script, i.e. the preprocessing pipeline __preprocess.py__ that currently does roughly this:

### Telespor (telespor)
- cleans telespor export for lat, lon, and batteryvoltage
- calculates speed and engine state (on/off) from telespor data
- resmaples the data to a sampling rate that could be set via arguments and interpolates missing lat lon, and batteryvoltage values

### Box of sensors (capture)
- adds absolute timestmps to the data using the recorded time of installation
- resamples (downsamples) the data to the same sampling rate as telespor measurements by averaging meaurements inside the bins

- For preparing training data, __preprocess.py__ also accepts two timestamps (signalstart & signalend) that limits the measurements used to the time range during which the boat was actually moving

## Classification

* Features: box measurements (ax, ay, az, gx, gy, gz)
* Labels: engine state (ON/OFF) or batteryvoltage (passed as a sys arg to classify.py)

### Weather (used)
- [AROME-arctic full model forecast](https://www.met.no/en/projects/The-weather-model-AROME-Arctic)
- [WAM4 archive](http://thredds.met.no/thredds/catalog/fou-hi/mywavewam4archive/catalog.html)

### Weather (for reference)
- [list of free MET data](https://www.met.no/frie-meteorologiske-data/frie-meteorologiske-data)
- [Latest YR forecast (including WAM forecasts)](https://www.met.no/frie-meteorologiske-data/frie-meteorologiske-data)
