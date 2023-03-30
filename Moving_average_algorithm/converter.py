# Exponential smoothing algorithm for paro data (barometer)
# Tristan Liang
# 1/19/2023

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import matplotlib
import time
import array
from datetime import datetime
from matplotlib.ticker import StrMethodFormatter
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import butter, lfilter, detrend, welch, spectrogram
from scipy.signal.windows import hamming

# Insturction !!
# 1st step: do the manual setting, where you can changes the value to what you want
# 2nd step: uncomment the plot function in the main function (only uncomment one at a time)

""""""""""""""""""""""""""""" manual setting """""""""""""""""""""""""""""""""
filename = 'westover-2-baro.csv' # select the datafile
starting_time = '2022-08-18 20:04:00' # any select time in this format
format_data = "20%y-%m-%d %H:%M:%S" # time format                       ==
number_of_samples = 6000 # number_of_samples = ? minutes * (1200 samples/minutes)
alpha = 0.03 # smoothing factor, 1 > alpha > 0
event_name = "HAWKER LANDING" # Name of the event, show n the title of the graph
p,d,q = 10,1,10 # pdq value of ARIMA model
test_peak_minute_after_starttime = 2.5 # minutes after the start time (= the window start time) for plotting maximum difference
A, B = 6, 2  # Fast Fourier transform (FFT) Window size: (2^A) and Shift size: (2^B)
dB = False # plot power in dB, True or False
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

## function of opening the csv data file and categorize it
def categorize_paros_datafile():

    file = open(filename)
    type(file)
    csvreader = csv.reader(file)

    ## categorize 6 barometers in two boxes into seperate lists
    paro141920 = [] ##create empty list for the paro 1's barometer #141920
    paro142180 = [] ##create empty list for the paro 1's barometer #142180
    paro141906 = [] ##create empty list for the paro 2's barometer #141906
    paro142176 = [] ##create empty list for the paro 2's barometer #142176
    paro141905 = [] ##create empty list for the paro 3's barometer #141905
    paro141931 = [] ##create empty list for the paro 3's barometer #141931

    for row in csvreader:
        if row[1] == str(141905): 
            paro141905.append(row)
        if row[1] == str(141931):
            paro141931.append(row)
        if row[1] == str(141906.0):
            paro141906.append(row)
        if row[1] == str(142176.0):
            paro142176.append(row)
        if row[1] == str(141920):
            paro141920.append(row)
        if row[1] == str(142180):
            paro142180.append(row)

    file.close()
    return paro141920, paro142180, paro141906 ,paro142176 , paro141905, paro141931

## function of finding the index of each barometer's stating time from the data list
def time_matched_index(parometer_model_data, starting_date):
    data = np.array(parometer_model_data)

    # Convert the datetime strings to NumPy datetime64 objects
    datetime_strings = np.array([x[2] for x in data])
    datetimes = np.array([np.datetime64(x).astype('datetime64[s]') for x in datetime_strings])
    
    # Convert the target time string to a datetime.datetime object
    target_time = datetime.strptime(starting_time, "%Y-%m-%d %H:%M:%S")

    # Convert the target time to a NumPy datetime64 object
    target_time_np = np.datetime64(target_time)
    
    # Find the closest datetime to the target time
    closest_datetime = datetimes[np.argmin(np.abs(datetimes - target_time_np))]
    
    # Find the index of the closest datetime in the original array
    index = np.where(datetimes == closest_datetime)[0][0]

    return index

## function of calculate the moving average (exponantial smoothing)
# Formula : https://en.wikipedia.org/wiki/Exponential_smoothing
def exp_smoothing_from_raw_data(time_matched_index, parometer_model_data):
    start_string = parometer_model_data[time_matched_index]
    start_pressure = start_string[3] # pressure of the starting time.
    raw=[] # raw data list
    exp_smooth = [] # create empty list for the exponential smoothing algorithm
    exp_smooth.append(start_pressure)
    raw.append(float(start_pressure))

    for i in range(1,number_of_samples):
        a = time_matched_index+ i
        x_t = parometer_model_data[a]
        raw.append(float(x_t[3]))
        predict = str(alpha * float(x_t[3]) + (1 - alpha) *float(exp_smooth[i-1]))
        exp_smooth.append(predict)

    return raw, exp_smooth

## function of creating matrix for the x axis for main graph
def matrix_x_axis():

    x_matrix = []
    for i in range(number_of_samples):
        x_matrix.append(i/1200)

    return x_matrix

## function of creating matrix for the y axis for main graph
def matrix_y_axis(raw, exp_smooth):

    y_rot_exp_smooth = [float(x) for x in exp_smooth]
    y_rot_raw = [float(x) for x in raw]

    return y_rot_exp_smooth, y_rot_raw

## function of creating matrix for the y axis for difference graph
def matrix_axis_diff(raw, exp_smooth):

    dx = 0.05 
    first_derivative = np.concatenate((np.array([0]), np.diff(raw)))/dx
    second_derivative = np.concatenate((np.array([0]), np.diff(first_derivative)))/dx
    y_der_1 = [float(x) for x in first_derivative]
    y_der_2 = [float(x) for x in second_derivative]
    y_diff = [float(x) - float(y) for x, y in zip(raw, exp_smooth)]

    return y_der_1, y_der_2, y_diff

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

    model = ARIMA(raw, order=(p,d,q))
    results_ARIMA = model.fit()
    y3_series=pd.Series(results_ARIMA.predict())
    y3 = np.roll(y3_series, -1) # move left by one sample

    return y3

# function of applying a time window then show the locations of maximum difference
def line4peak(raw_data):
    index_list = []
    window_start_index = int(0 + test_peak_minute_after_starttime*1200)
    window_end_index = int(99 + test_peak_minute_after_starttime*1200)
    for i in range(12):
        max_diff_index = None
        max_diff = float(0)
        for f in range(window_start_index, window_end_index):
            diff = abs(raw_data[f+1] - raw_data[f])
            if diff > max_diff:
                max_diff = diff
                max_diff_index = f
        index_list.append(max_diff_index)
        window_start_index = window_start_index + 100
        window_end_index = window_end_index + 100
    return index_list

def process_segment(data_segment):

    # Apply the FFT to the data segment
    fft_output = np.fft.fft(data_segment)
    # Take the absolute value (magnitude) of the FFT output
    magnitude = np.abs(fft_output)
    squared_magnitude = np.square(magnitude)
    # Sum the magnitude values of the FFT output
    sum_magnitude = np.sum(squared_magnitude[:len(squared_magnitude)//2])
    # Return the sum of magnitudes
    return float(sum_magnitude)

def gen_power_data(raw, exp):
    fft_window_size = 2**A

    # Process each segment of data
    results = []
    for i in range(0, len(raw)-fft_window_size+1):
        data_segment = [raw[i] - float(exp[i]) for i in range(fft_window_size)]
        result = process_segment(data_segment)
        results.append(float(result))
    # add None at the end of the results list(y axis) in order to match the size of x axis
    results += [None] * (fft_window_size - 1)
    # convert list to array
    result_array = [ x for x in results]
    
    return result_array

## function of calculating the FFT of the residual(raw - ES)
def fft_ac(raw, exp):

    data = [raw[i] - float(exp[i]) for i in range(len(raw))]
    
    # Calculate the number of windows
    n_windows = int((len(data) - 2**A) / 2**B) + 1
    
    # Create an empty array to store the squared sums
    squared_sums = np.zeros(n_windows)
    # Loop through each window

    for i in range(n_windows):
        # Extract the current window from the signal
        window = data[i*2**B:i*2**B+2**A]
        
        # Apply FFT to the window
        fft = np.fft.fft(window)
        
        # Extract the positive frequencies
        positive_freqs = fft[:2**A//2]
        
        # Take the absolute values
        abs_values = np.abs(positive_freqs)
        
        # Calculate the squared sum
        squared_sums[i] = np.sum(abs_values)**2

    # covert to dB
    if dB == True:
        db_sum = 10*np.log10(squared_sums)
        result_array = [ x for x in db_sum]
    else:
        result_array = [ x for x in squared_sums]


        
    # Return the squared sums for each window
    return result_array

## function of plotting the FFT of the residual(raw - ES)
def gen_fft_ac_graph(raw_1, exp_smooth_1, raw_2, exp_smooth_2, raw_3, exp_smooth_3, raw_4, exp_smooth_4, raw_5, exp_smooth_5, raw_6, exp_smooth_6, starting_date):
    
    fft_window_size = 2**A
    x_matrix = []
    y1, y2, y3, y4, y5, y6 = (fft_ac(raw_1, exp_smooth_1)), fft_ac(raw_2, exp_smooth_2),fft_ac(raw_3, exp_smooth_3),fft_ac(raw_4, exp_smooth_4),fft_ac(raw_5, exp_smooth_5),fft_ac(raw_6, exp_smooth_6)
    number_of_windows = len(y1)
    for i in range(number_of_windows):
        x_matrix.append(i/1200)
    x = x_matrix
    fig_main = plt.figure(figsize=(15, 10))
    st = "%s/%s/%s %s:%s"%(starting_date.year, starting_date.month, starting_date.day, starting_date.hour, starting_date.minute)
    fig_main.suptitle("%s -- Power (magnitude of the sum of the FFT) \n (starting time is %s:00)"%( event_name, st), fontsize=20, fontweight = 'bold')
    fig_main.supxlabel("Timestamp (UTC)", fontsize=20,fontweight = 'bold')
    if dB == True:
        fig_main.supylabel("Power (Pa/Hz) in dB", fontsize=20, fontweight = 'bold')
    else:
        fig_main.supylabel("Power (Pa/Hz)", fontsize=20, fontweight = 'bold')
    title_1 = 'Sum of the FFT in the window of %s samples shift by %s'%(fft_window_size,2**B)
    device_name_1 = "paro1-141920"
    device_name_2 = "paros1-142180"
    device_name_3 = "paro2-141906"
    device_name_4 = "paro2-142176"
    device_name_5 = "paro3-141905"
    device_name_6 = "paro3-141931"


    plt.subplot(6, 1, 1)
    plt.plot(x, y1,label = title_1, linewidth=1, color = 'blue')
    plt.text(0.1, 0.1, device_name_1, ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    plt.xticks([])
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)


    plt.subplot(6, 1, 2)
    plt.plot(x, y2, label = title_1, linewidth=1, color = 'green')
    plt.text(0.1, 0.1, device_name_2, ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    plt.xticks([])  


    plt.subplot(6, 1, 3)
    plt.plot(x, y3, label = title_1, linewidth=1, color = 'orange')
    plt.text(0.1, 0.1, device_name_3, ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    plt.xticks([])  


    plt.subplot(6, 1, 4)
    plt.plot(x, y4, label = title_1, linewidth=1, color = 'brown')
    plt.text(0.1, 0.1, device_name_4, ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    plt.xticks([])  
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)

    plt.subplot(6, 1, 5)
    plt.plot(x, y5, label = title_1, linewidth=1, color = 'red')
    plt.text(0.1, 0.1, device_name_5, ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    plt.xticks([])  
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)

    startnum = 1000*(starting_date.hour*60 + starting_date.minute)
    endnum = startnum + number_of_samples/1.2
    step = number_of_windows
    segtime = []
    for i in range(step):
        nn = (endnum - startnum)/step
        segtime.append(startnum + nn*i)
    x = segtime

    plt.subplot(6, 1, 6)
    plt.plot(x, y6, label = title_1, linewidth=1, color = 'magenta')
    plt.text(0.1, 0.1, device_name_6, ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    startnum = 1000*(starting_date.hour*60 + starting_date.minute)
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, framealpha=1)
    plt.xticks(fontsize=16)
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 1000)))
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.show()

    return()

## function of plotting the raw data, exp smoothing, and ARIMA prediction
def gen_main_graphs(raw_1, exp_smooth_1, raw_2, exp_smooth_2, raw_3, exp_smooth_3, raw_4, exp_smooth_4, raw_5, exp_smooth_5, raw_6, exp_smooth_6, starting_date):
    fig_main = plt.figure(figsize=(15, 10))
    x = matrix_x_axis()
    st = "%s/%s/%s %s:%s"%(starting_date.year, starting_date.month, starting_date.day, starting_date.hour, starting_date.minute)
    fig_main.suptitle("%s -- Raw vs ARIMA prediction vs Exponential Smoothing \n (starting time is %s:00)"%(event_name, st), fontsize=26, fontweight = 'bold')
    fig_main.supxlabel("Timestamp (UTC)", fontsize=20,fontweight = 'bold')
    fig_main.supylabel("Pressure (hPa)", fontsize=20, fontweight = 'bold')
    title_1 = 'ARIMA (p,d,q = %s,%s,%s)'%(p,d,q)
    title_2 = 'E.S. (alpha = %s)'%(alpha)
    device_name_1 = "paro1-141920"
    device_name_2 = "paros1-142180"
    device_name_3 = "paro2-141906"
    device_name_4 = "paro2-142176"
    device_name_5 = "paro3-141905"
    device_name_6 = "paro3-141931"

    # finding when the maximum difference in each five seconds window for each barometer raw data
    line1, line2, line3, line4, line5, line6 = line4peak(raw_1), line4peak(raw_2), line4peak(raw_3), line4peak(raw_4), line4peak(raw_5), line4peak(raw_6)


    y1, y2= matrix_y_axis(raw_1, exp_smooth_1)
    y3 = arima_model(raw_1)
    plt.subplot(6, 1, 1)
    plt.plot(x, y2, label='raw - %s'%(device_name_1), linewidth=0.8, color = 'black')
    plt.plot(x, y3, label = title_1, linewidth= 0.8, color = 'grey')
    plt.plot(x, y1,label = title_2, linewidth=1, color = 'blue')
    for i in line1:
        plt.axvline(i/1200, color = 'b',linewidth = 0.5)
    plt.xticks([])
    plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    y1, y2= matrix_y_axis(raw_2, exp_smooth_2)
    y3 = arima_model(raw_2)
    plt.subplot(6, 1, 2)
    plt.plot(x, y2, label ='raw - %s'%(device_name_2), linewidth=0.8, color = 'black')
    plt.plot(x, y3, label = title_1, linewidth= 0.8, color = 'grey')
    plt.plot(x, y1, label = title_2, linewidth=1, color = 'green')
    for i in line2:
        plt.axvline(i/1200, color = 'b',linewidth = 0.5)
    plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.xticks([])  
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    y1, y2= matrix_y_axis(raw_3, exp_smooth_3)
    y3 = arima_model(raw_3)
    plt.subplot(6, 1, 3)
    plt.plot(x, y2, label ='raw - %s'%(device_name_3), linewidth=0.8, color = 'black')
    plt.plot(x, y3, label = title_1, linewidth= 0.8, color = 'grey')
    plt.plot(x, y1, label = title_2, linewidth=1, color = 'orange')
    for i in line3:
        plt.axvline(i/1200, color = 'b',linewidth = 0.5)
    plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.xticks([])
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    y1, y2= matrix_y_axis(raw_4, exp_smooth_4)
    y3 = arima_model(raw_4)
    plt.subplot(6, 1, 4)
    plt.plot(x, y2, label ='raw - %s'%(device_name_4), linewidth=0.8, color = 'black')
    plt.plot(x, y3, label = title_1, linewidth= 0.8, color = 'grey')
    plt.plot(x, y1, label = title_2, linewidth=1, color = 'brown')
    for i in line4:
        plt.axvline(i/1200, color = 'b',linewidth = 0.5)
    plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.xticks([])  
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    y1, y2= matrix_y_axis(raw_5, exp_smooth_5)
    y3 = arima_model(raw_5)
    plt.subplot(6, 1, 5)
    plt.plot(x, y2, label ='raw - %s'%(device_name_5), linewidth=0.8, color = 'black')
    plt.plot(x, y3, label = title_1, linewidth= 0.8, color = 'grey')
    plt.plot(x, y1, label = title_2, linewidth=1, color = 'red')
    for i in line5:
        plt.axvline(i/1200, color = 'b',linewidth = 0.5)
    plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.xticks([])  
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    y1, y2= matrix_y_axis(raw_6, exp_smooth_6)
    y3 = arima_model(raw_6)
    x = x_axis_tick(starting_date)
    plt.subplot(6, 1, 6)
    plt.plot(x, y2, label ='raw - %s'%(device_name_6), linewidth=0.8, color = 'black')
    plt.plot(x, y3, label = title_1, linewidth= 0.8, color = 'grey')
    plt.plot(x, y1,label = title_2, linewidth=1, color = 'magenta')
    startnum = 1000*(starting_date.hour*60 + starting_date.minute)
    for i in line4:
        plt.axvline(startnum + i/1.2, color = 'b', linewidth = 0.5)
    plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, framealpha=1)
    plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.xticks(fontsize=16)
    plt.ticklabel_format(useOffset=False, style='plain')
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 1000)))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    
    plt.show()

    return()

## function of plotting the raw data, exp smoothing, and ARIMA prediction for single paro
def gen_main_graphs_single(raw_1, exp_smooth_1, raw_2, exp_smooth_2, starting_date):
    fig_main = plt.figure(figsize=(15, 10))
    x = matrix_x_axis()
    st = "%s/%s/%s %s:%s"%(starting_date.year, starting_date.month, starting_date.day, starting_date.hour, starting_date.minute)
    fig_main.suptitle("%s -- Raw vs ARIMA prediction vs Exponential Smoothing \n (starting time is %s:00)"%(event_name, st), fontsize=26, fontweight = 'bold')
    fig_main.supxlabel("Timestamp (UTC)", fontsize=20,fontweight = 'bold')
    fig_main.supylabel("Pressure (hPa)", fontsize=20, fontweight = 'bold')
    title_1 = 'ARIMA (p,d,q = %s,%s,%s)'%(p,d,q)
    title_2 = 'E.S. (alpha = %s)'%(alpha)
    device_name_1 = "paros2-141906"
    device_name_2 = 'paros2-142176'

    # finding when the maximum difference in each five seconds window for each barometer raw data
    line1, line2=  line4peak(raw_1), line4peak(raw_1)

    y1, y2= matrix_y_axis(raw_1, exp_smooth_1)
    y3 = arima_model(raw_1)
    plt.subplot(2, 1, 1)
    plt.plot(x, y2, label ='raw - %s'%(device_name_1), linewidth=0.8, color = 'black')
    plt.plot(x, y3, label = title_1, linewidth= 0.8, color = 'grey')
    plt.plot(x, y1, label = title_2, linewidth=1, color = 'orange')
    for i in line1:
        plt.axvline(i/1200, color = 'b',linewidth = 0.5)
    plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.xticks([])
    plt.legend(fontsize=15, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    y1, y2= matrix_y_axis(raw_1, exp_smooth_2)
    y3 = arima_model(raw_2)
    x = x_axis_tick(starting_date)
    plt.subplot(2, 1, 2)
    plt.plot(x, y2, label ='raw - %s'%(device_name_2), linewidth=0.8, color = 'black')
    plt.plot(x, y3, label = title_1, linewidth= 0.8, color = 'grey')
    plt.plot(x, y1, label = title_2, linewidth=1, color = 'brown')
    startnum = 1000*(starting_date.hour*60 + starting_date.minute)
    for i in line2:
        plt.axvline(startnum + i/1.2, color = 'b', linewidth = 0.5)
    plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.legend(fontsize=15, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, framealpha=1)
    plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.xticks(fontsize=16)
    plt.ticklabel_format(useOffset=False, style='plain')
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 1000)))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    

    plt.show()
    return()

## function of plotting the residual (raw - ES), first derivative, second derivative
def gen_diff_graphs(raw_1, exp_smooth_1, raw_2, exp_smooth_2, raw_3, exp_smooth_3, raw_4, exp_smooth_4, raw_5, exp_smooth_5, raw_6, exp_smooth_6, starting_date):
    
    fig = plt.figure(figsize=(15, 10))
    x = matrix_x_axis()
    st = "%s/%s/%s %s:%s"%(starting_date.year, starting_date.month, starting_date.day, starting_date.hour, starting_date.minute)
    fig.suptitle("%s -- Residuals (raw - E.S.) and Derivative of Raw Data\n (starting time is %s:00)"%(event_name, st), fontsize=26, fontweight = 'bold')
    fig.supxlabel("Timestamp (UTC)", fontsize=20,fontweight = 'bold')
    fig.supylabel("Pressure (hPa)", fontsize=20, fontweight = 'bold')
    title_1 = 'First Derivative'
    title_2 = 'Second Derivative'
    title_3 = 'Residuals (raw - E.S.)'

    y1, y2, y3= matrix_axis_diff(raw_1, exp_smooth_1)
    plt.subplot(6, 1, 1)
    plt.plot(x, y1, label = title_1, linewidth=1, color = "blue") 
    plt.plot(x, y2, label = title_2, linewidth=1, color = "green")
    plt.plot(x, y3, label = title_3, linewidth=1, color = "orange")
    plt.text(0.1, 0.1, "paro1 - 141920", ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    plt.xticks([])
    # plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    # plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    y1, y2, y3= matrix_axis_diff(raw_2, exp_smooth_2)
    plt.subplot(6, 1, 2)
    plt.plot(x, y1, label = title_1, linewidth=1, color = "blue") 
    plt.plot(x, y2, label = title_2, linewidth=1, color = "green")
    plt.plot(x, y3, label = title_3, linewidth=1, color = "orange")
    plt.text(0.1, 0.1, "paro1 - 142180", ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    # plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.xticks([])  
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    # plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    y1, y2, y3= matrix_axis_diff(raw_3, exp_smooth_3)
    plt.subplot(6, 1, 3)
    plt.plot(x, y1, label = title_1, linewidth=1, color = "blue") 
    plt.plot(x, y2, label = title_2, linewidth=1, color = "green")
    plt.plot(x, y3, label = title_3, linewidth=1, color = "orange")
    plt.text(0.1, 0.1, "paro2 - 141906", ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    # plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.xticks([])  
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    # plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    y1, y2, y3= matrix_axis_diff(raw_4, exp_smooth_4)
    plt.subplot(6, 1, 4)
    plt.plot(x, y1, label = title_1, linewidth=1, color = "blue") 
    plt.plot(x, y2, label = title_2, linewidth=1, color = "green")
    plt.plot(x, y3, label = title_3, linewidth=1, color = "orange")
    plt.text(0.1, 0.1, "paro2 - 142176", ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    # plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    plt.xticks([]) 
    # plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    y1, y2, y3= matrix_axis_diff(raw_5, exp_smooth_5)
    plt.subplot(6, 1, 5)
    plt.plot(x, y1, label = title_1, linewidth=1, color = "blue") 
    plt.plot(x, y2, label = title_2, linewidth=1, color = "green")
    plt.plot(x, y3, label = title_3, linewidth=1, color = "orange")
    plt.text(0.1, 0.1, "paro3 - 141905", ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    # plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.xticks([])  
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    # plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    y1, y2, y3= matrix_axis_diff(raw_6, exp_smooth_6)
    x = x_axis_tick(starting_date)
    plt.subplot(6, 1, 6)
    plt.plot(x, y1, label = title_1, linewidth=1, color = "blue") 
    plt.plot(x, y2, label = title_2, linewidth=1, color = "green")
    plt.plot(x, y3, label = title_3, linewidth=1, color = "orange")
    plt.text(0.1, 0.1, "paro3 - 141931", ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    # plt.yticks(np.arange(min(y2), max(y2), 0.2), fontsize=12)
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, framealpha=1)
    # plt.ylim(min(y2)-0.02, max(y2)+0.02)
    plt.xticks(fontsize=16)
    plt.ticklabel_format(useOffset=False, style='plain')
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 1000)))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    

    plt.show()
    return()

## function of plotting the residual (raw - ES), first derivative, second derivative for single paro
def gen_diff_graphs_single(raw_1, exp_smooth_1, raw_2, exp_smooth_2, starting_date):
    
    fig = plt.figure(figsize=(15, 10))
    x = matrix_x_axis()
    st = "%s/%s/%s %s:%s"%(starting_date.year, starting_date.month, starting_date.day, starting_date.hour, starting_date.minute)
    fig.suptitle("%s -- Residuals (raw - E.S.) and Derivative of Raw Data\n (starting time is %s:00)"%(event_name, st), fontsize=26, fontweight = 'bold')
    fig.supxlabel("Timestamp (UTC)", fontsize=20,fontweight = 'bold')
    fig.supylabel("Pressure (hPa)", fontsize=20, fontweight = 'bold')
    title_1 = 'First Derivative'
    title_2 = 'Second Derivative'
    title_3 = 'Residuals (raw - E.S.)'
    device_name_1 = "paro2 - 141906"
    device_name_2 = "paro2 - 142176"


    y1, y2, y3= matrix_axis_diff(raw_1, exp_smooth_1)
    plt.subplot(2, 1, 1)
    plt.plot(x, y1, label = title_1, linewidth=1, color = "blue")  # plot First Derivative
    # plt.plot(x, y2, label = title_2, linewidth=1, color = "green") # plot Second Derivative
    plt.plot(x, y3, label = title_3, linewidth=1, color = "orange") # plot Residuals (raw - E.S.)
    plt.text(0.1, 0.1, device_name_1, ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    plt.xticks([])  
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    y1, y2, y3= matrix_axis_diff(raw_2, exp_smooth_2)
    x = x_axis_tick(starting_date)
    plt.subplot(2, 1, 2)
    plt.plot(x, y1, label = title_1, linewidth=1, color = "blue") 
    # plt.plot(x, y2, label = title_2, linewidth=1, color = "green")
    plt.plot(x, y3, label = title_3, linewidth=1, color = "orange")
    plt.text(0.1, 0.1, device_name_2, ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    plt.xticks(fontsize=16)
    plt.ticklabel_format(useOffset=False, style='plain')
    formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 1000)))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    plt.show()
    return()

## function of plotting PSD
def gen_spetrogram(raw):

    # raw *= 100  # convert to Pa from hPa
    raw  *= 100

    welchB = 256
    welch0 = 32
    w_window = hamming(welchB)

    raw_array = np.array(raw)
    #f, t, Sxx = spectrogram(df[device], fs=20, window=w_window, noverlap=welch0, nfft=NFFT, return_onesided=True, mode='psd')
    f, t, Sxx = spectrogram(raw_array, fs=20, window=w_window, noverlap=welch0, nfft=welchB, return_onesided=True, mode='psd')

        
    # convert to timestamps (from seconds)
    """t_timestamps = []
    for i in t:
        t_timestamps.append(datetime.timedelta(seconds=i) + start_time)"""
    # t = x_axis_tick(starting_date)

    plt.figure()

    plt.pcolormesh(t, f, np.log10(Sxx), shading='gouraud')
    plt.title(event_name, fontsize=22)
    plt.ylabel('Frequency (Hz)', fontsize=18)
    plt.xlabel('Timestamp (UTC)', fontsize=18)
    plt.colorbar(label='Pa^2/Hz')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():

    # constant
    starting_date = datetime.strptime(starting_time, format_data)
    paro1_rows1, paro1_rows2, paro2_rows1, paro2_rows2, paro3_rows1, paro3_rows2 = categorize_paros_datafile()
    
    ## find the index of each barometer the stating time from the list
    time_matched_index_141920 = time_matched_index(paro1_rows1, starting_date)
    time_matched_index_142180 = time_matched_index(paro1_rows2, starting_date)
    time_matched_index_141906 = time_matched_index(paro2_rows1, starting_date)
    time_matched_index_142176 = time_matched_index(paro2_rows2, starting_date)
    time_matched_index_141905 = time_matched_index(paro3_rows1, starting_date)
    time_matched_index_141931 = time_matched_index(paro3_rows2, starting_date)

    ## create the list of pressure of raw data and exponential smoothing data
    raw_141920, exp_smooth_141920 = exp_smoothing_from_raw_data(time_matched_index_141920, paro1_rows1)
    raw_142180, exp_smooth_142180 = exp_smoothing_from_raw_data(time_matched_index_142180, paro1_rows2)
    raw_141906, exp_smooth_141906 = exp_smoothing_from_raw_data(time_matched_index_141906, paro2_rows1)
    raw_142176, exp_smooth_142176 = exp_smoothing_from_raw_data(time_matched_index_142176, paro2_rows2)
    raw_141905, exp_smooth_141905 = exp_smoothing_from_raw_data(time_matched_index_141905, paro3_rows1)
    raw_141931, exp_smooth_141931 = exp_smoothing_from_raw_data(time_matched_index_141931, paro3_rows2)

    #generate plot for raw data, exp smoothing, and ARIMA prediction for all paros
    # gen_main_graphs(raw_141920, exp_smooth_141920, raw_142180, exp_smooth_142180, raw_141906, exp_smooth_141906, raw_142176, exp_smooth_142176, raw_141905, exp_smooth_141905, raw_141931, exp_smooth_141931, starting_date)
        
    # generate plot for raw - ES, first derivative, second derivative for single paro
    # gen_main_graphs_single(raw_141906, exp_smooth_141906, raw_142176, exp_smooth_142176, starting_date)

    # generate plot for raw - ES, first derivative, second derivative for all paros
    # gen_diff_graphs(raw_141920, exp_smooth_141920, raw_142180, exp_smooth_142180, raw_141906, exp_smooth_141906, raw_142176, exp_smooth_142176, raw_141905, exp_smooth_141905, raw_141931, exp_smooth_141931, starting_date)

    # generate plot for raw - ES, first derivative, second derivative for single paro
    # gen_diff_graphs_single(raw_141906, exp_smooth_141906, raw_142176, exp_smooth_142176, starting_date)

    # generate spectrogram (not finished)
    # gen_spetrogram(raw_141906)

    # generate power fft over time
    gen_fft_ac_graph(raw_141920, exp_smooth_141920, raw_142180, exp_smooth_142180, raw_141906, exp_smooth_141906, raw_142176, exp_smooth_142176, raw_141905, exp_smooth_141905, raw_141931, exp_smooth_141931, starting_date)

    return()

if __name__ == "__main__":
    main()