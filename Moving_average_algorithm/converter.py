# Exponential smoothing algorithm for paro data (barometer)
# Tristan Liang
# 1/19/2023

import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import matplotlib
import statsmodels
import statsmodels.api as sm
from turtle import end_fill
from datetime import datetime
from pylab import rcParams
from matplotlib.ticker import StrMethodFormatter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

""""""""""""""""""""""""""""" manual setting """""""""""""""""""""""""""""""""
# select the file, starting time, number_of_samples and alpha 
filename = '20220816-0946.csv' # select the datafile
starting_time = '08/16/22 17:32:24.625955' # any select time in this format
number_of_samples = 6000 # number_of_samples = ? minutes * (1200 samples/minutes)
alpha = 0.01 # smoothing factor, 1 > alpha > 0
event_name = "Sample 1" # Name of the event, show n the title of the graph
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

## function of opening the csv data file and categorize it
def categorize_paros_datafile(filename):

    #file = open('20220816-0946.csv')
    file = open(filename)
    type(file)
    csvreader = csv.reader(file)

    ## categorize 4 barometers in two boxes in seperate lists
    paro141905 = [] ##create empty list for the paro 3's barometer #141905
    paro141931 = [] ##create empty list for the paro 3's barometer #141931
    paro141920 = [] ##create empty list for the paro 1's barometer #141920
    paro142180 = [] ##create empty list for the paro 1's barometer #142180

    for row in csvreader:
        if row[1] == str(141905): 
            paro141905.append(row)
        if row[1] == str(141931):
            paro141931.append(row)
        if row[1] == str(141920):
            paro141920.append(row)
        if row[1] == str(142180):
            paro142180.append(row)

    file.close()
    return paro141905, paro141931, paro141920, paro142180

## function of finding the index of each barometer's stating time from the data list
def time_matched_index(parometer_model_data, starting_date):
    # barometer 141905
    time_matched_index = [] 
    i = 0
    length = len(parometer_model_data) # select which barometer
    while i < length:
        string_i = parometer_model_data[i]
        paro_string = string_i[2]
        paro_date = datetime.strptime(paro_string, format_data)
        # store the index of the time in datasheet that match the stating time (accurate to second)
        if  (starting_date.year == paro_date.year) and (starting_date.month == paro_date.month) and (starting_date.day == paro_date.day) and (starting_date.hour == paro_date.hour) and (starting_date.minute == paro_date.minute):
            time_matched_index.append(i)
            break
        i += 1
    return time_matched_index

## calculate the moving average (exponantial smoothing)
# Formula : https://en.wikipedia.org/wiki/Exponential_smoothing
def exp_smoothing_from_raw_data(time_matched_index, parometer_model_data, number_of_samples, alpha):
    start_string = parometer_model_data[time_matched_index[0]]
    start_pressure = start_string[3] # pressure of the starting time.
    raw=[] # raw data list
    exp_smooth = [] # create empty list for the exponential smoothing algorithm
    exp_smooth.append(start_pressure)
    raw.append(float(start_pressure))

    for i in range(1,number_of_samples):
        a = time_matched_index[0] + i
        x_t = parometer_model_data[a]
        raw.append(float(x_t[3]))
        predict = str(alpha * float(x_t[3]) + (1 - alpha) *float(exp_smooth[i-1]))
        exp_smooth.append(predict)

    return raw, exp_smooth

## function of creating matrix for the x and y axis
def matrix_axis(raw, exp_smooth, number_of_samples):
    x_matrix = []
    for i in range(number_of_samples):
        x_matrix.append(i/1200)
    y_rot_exp_smooth = [float(x) for x in exp_smooth]
    y_rot_raw = [float(x) for x in raw]

    return y_rot_exp_smooth, y_rot_raw, x_matrix

## function of creating datasheet for real time x axis tick label
def x_axis_tick(starting_date):
    startnum = 1000*(starting_date.hour*60 + starting_date.minute)
    endnum = startnum + number_of_samples/1.2
    step = number_of_samples

    segtime = []
    for i in range(step):
        nn = (endnum - startnum)/step
        segtime.append(startnum + nn*i)

    return segtime

## function of ARIMA model
def arima_model(raw):

    """
    ts_log_array = np.log(raw) # making Series Stationary
    ts_log = pd.Series(ts_log_array) 
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff = ts_log_diff[~ts_log_diff.isnull()]
    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20)
    model = ARIMA(ts_log_diff, order=(1,1,1))  # order = (p,d,q)
    model = ARIMA(raw, order=(1,1,1))
    results_ARIMA = model.fit()
    ARIMA_diff_predictions = pd.Series(results_ARIMA.fittedvalues, copy=True)
    ARIMA_diff_predictions_cumsum = ARIMA_diff_predictions.cumsum()
    ARIMA_log_prediction = pd.Series(ts_log.iloc[0], index=ts_log.index)
    ARIMA_log_prediction = ARIMA_log_prediction.add(ARIMA_diff_predictions_cumsum,fill_value=0)
    ARIMA_log_prediction.head()
    predictions_ARIMA = np.exp(ARIMA_log_prediction)
    #print(np.shape(predictions_ARIMA))
    #print(type(predictions_ARIMA))
    """

    """
    X = raw
    size = int(len(X) * 0.66)
    predictions = list()
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]

    for t in range(len(test)):
        model = ARIMA(history, order=(1,1,1))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        # print('predicted=%f, expected=%f' % (yhat, obs))
    """

    model = ARIMA(raw, order=(1,1,1))
    results_ARIMA = model.fit()

    # finding the p, d, q value
    """fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts_log_diff.dropna(),lags=40,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts_log_diff.dropna(),lags=40,ax=ax2)
    plt.show()"""

    return results_ARIMA

## function of plotting
def gen_graphs(raw_1, exp_smooth_1, raw_2, exp_smooth_2, raw_3, exp_smooth_3, raw_4, exp_smooth_4, starting_date):
    fig = plt.figure(figsize=(15, 10))
    st = "%s/%s/%s %s:%s"%(starting_date.year, starting_date.month, starting_date.day, starting_date.hour, starting_date.minute)
    fig.suptitle("%s -- Raw vs Exponential_smoothing \n (starting time is %s:00)"%(event_name, st), fontsize=28, fontweight = 'bold')
    fig.supxlabel("Timestamp (UTC)", fontsize=24,fontweight = 'bold')
    fig.supylabel("Pressure (hPa)", fontsize=24, fontweight = 'bold')


    plt.subplot(4, 1, 1)
    y1, y2, x = matrix_axis(raw_1, exp_smooth_1, number_of_samples)
    pred = arima_model(raw_1)
    y3_series=pd.Series(pred.predict()) # Create a Pandas Series from array
    y3 = np.roll(y3_series, -1) # move left by one sample
    plt.plot(x, y2, label='raw - paros3-141905', linewidth=0.5, color = 'black')
    plt.plot(x, y3, label ='Prediction from ARIMA model', linewidth= 0.5, color = 'grey')
    plt.plot(x, y1,label='Exponential Smoothing (alpha = %s)'%(alpha), linewidth=1, color = 'blue')
    plt.xticks([])
    plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=16)
    plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.legend(fontsize=10)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    plt.subplot(4, 1, 2)
    y1, y2, x= matrix_axis(raw_2, exp_smooth_2, number_of_samples)
    pred = arima_model(raw_2)
    y3_series=pd.Series(pred.predict()) # Create a Pandas Series from array
    y3 = np.roll(y3_series, -1) # move left by one sample
    plt.plot(x, y2, label='raw - paros3-141931', linewidth=0.5, color = 'black')
    plt.plot(x, y3, label ='Prediction from ARIMA model', linewidth= 0.5, color = 'grey')
    plt.plot(x, y1, label='Exponential Smoothing (alpha = %s)'%(alpha), linewidth=1, color = 'green')
    plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=16)
    plt.xticks([])  
    plt.legend(fontsize=10)
    plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    
    plt.subplot(4, 1, 3)
    y1, y2, x= matrix_axis(raw_3, exp_smooth_3, number_of_samples)
    pred = arima_model(raw_3)
    y3_series=pd.Series(pred.predict()) # Create a Pandas Series from array
    y3 = np.roll(y3_series, -1) # move left by one sample
    plt.plot(x, y2, label='raw - paros1-141920', linewidth=0.5, color = 'black')
    plt.plot(x, y3, label ='Prediction from ARIMA model', linewidth= 0.5, color = 'grey')
    plt.plot(x, y1, label='Exponential Smoothing (alpha = %s)'%(alpha), linewidth=1, color = 'red')
    plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=16)
    plt.xticks([])  
    plt.legend(fontsize=10)
    plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    plt.subplot(4, 1, 4)
    y1, y2, x= matrix_axis(raw_4, exp_smooth_4, number_of_samples)
    x = x_axis_tick(starting_date)
    pred = arima_model(raw_4)
    y3_series=pd.Series(pred.predict()) # Create a Pandas Series from array
    y3 = np.roll(y3_series, -1) # move left by one sample
    plt.plot(x, y2, label='raw - paros1-142180', linewidth=0.5, color = 'black')
    plt.plot(x, y3, label ='Prediction from ARIMA model', linewidth= 0.5, color = 'grey')
    plt.plot(x, y1,label='Exponential Smoothing (alpha = %s)'%(alpha), linewidth=1, color = 'magenta')
    plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=16)
    plt.legend(fontsize=10)
    plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.xticks(fontsize=16)
    plt.ticklabel_format(useOffset=False, style='plain')
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 1000)))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    

    plt.show()
    return()

def main():

    # constant
    global format_data
    format_data = "%m/%d/%y %H:%M:%S.%f"
    starting_date = datetime.strptime(starting_time, format_data)
    parometer3_rows1, parometer3_rows2, parometer1_rows1, parometer1_rows2 = categorize_paros_datafile(filename)
    
    ## find the index of each barometer the stating time from the list
    time_matched_index_141905 = time_matched_index(parometer3_rows1, starting_date)
    time_matched_index_141931 = time_matched_index(parometer3_rows2, starting_date)
    time_matched_index_141920 = time_matched_index(parometer1_rows1, starting_date)
    time_matched_index_142180 = time_matched_index(parometer1_rows2, starting_date)

    ## create the list of pressure of raw data and exponential smoothing data
    raw_141905, exp_smooth_141905 = exp_smoothing_from_raw_data(time_matched_index_141905, parometer3_rows1, number_of_samples, alpha)
    raw_141931, exp_smooth_141931 = exp_smoothing_from_raw_data(time_matched_index_141931, parometer3_rows2, number_of_samples, alpha)
    raw_141920, exp_smooth_141920 = exp_smoothing_from_raw_data(time_matched_index_141920, parometer1_rows1, number_of_samples, alpha)
    raw_142180, exp_smooth_142180 = exp_smoothing_from_raw_data(time_matched_index_142180, parometer1_rows2, number_of_samples, alpha)
    
    ## plot
    gen_graphs(raw_141905, exp_smooth_141905, raw_141931, exp_smooth_141931, raw_141920, exp_smooth_141920, raw_142180, exp_smooth_142180, starting_date)

    return()

if __name__ == "__main__":
    main()