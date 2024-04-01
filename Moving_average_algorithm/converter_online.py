# Exponential smoothing algorithm for paro data (barometer)
# Tristan Liang
# 1/19/2023

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.ticker import StrMethodFormatter
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import butter, lfilter, detrend, welch, spectrogram
from scipy.signal.windows import hamming
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta
from apiikey_access import api_key
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import matplotlib.dates as mdates
import os
from pathlib import Path
from io import StringIO

# Insturction !!
# 1st step: do the manual setting, where you can changes the value to what you want

""""""""""""""""""""""""""""" manual setting """""""""""""""""""""""""""""""""
"""Earthquake data download for matlab analysis"""
start_time = "2024-04-01T06:41:25" # the starting time in the plot
box = "parost2" # whi paros box (example: parost, parost2)
sensor_id = '141929' # barometer id (example: 141920 for parost, 141929 for parost2)
box_hz = 20 # 20 Hz is the sampling rate of the barometer
sample_time = 30 # miuntes, time duration of the plot
sample_rate = 1200 # sample/min, sampling rate of the plot, set 1~1200
save_file_name = 'e17t2.txt' # name of save data file
"""""""""""""""""""""""""""""""""""""""""""""""""""
edit the  above part for downloading data from influx
"""""""""""""""""""""""""""""""""""""""""""""""""""
event_name = "2024 earthquake" # Name of the event, show in the title of the graph
format_data = "20%y-%m-%dT%H:%M:%S" # time format of start_time
timestamp_line = True # add vertical line at specified time, True or False
distance_epic = 6.32*1.6 # in KM, for vertical line at specified date
alpha = 0.03 # smoothing factor of exponential smoothing, 1 > alpha > 0
p,d,q = 10,1,10 # p, d, q value of ARIMA model
A, B = 4, 3  # Fast Fourier transform (FFT) Window size: (2^A) and Shift size: (2^B)
dB = True # plot power in dB, True or False
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


""""""""""""""""""""""""""""" api_token setting """""""""""""""""""""""""""""""""
influxdb_apikey = api_key() # download your api_token from influxdB. Remember not to save it in the public file!
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

device_name = box + "-" + sensor_id
starting_date = datetime.strptime(start_time, format_data)
number_of_samples = sample_rate*sample_time
# Convert start_time from string to datetime object
start_time_obj = datetime.fromisoformat(start_time)
# Calculate end_time by adding sample_time (in minutes) to start_time
end_time_obj = start_time_obj + timedelta(minutes=sample_time)
# Convert end_time from datetime object to string in the same format as start_time
end_time = end_time_obj.isoformat()
speed_sound = 20.58 # KM/min
# Calculate the time taken in minutes
min_value = distance_epic / speed_sound
# Split the minutes into whole minutes and remaining seconds
whole_minutes = int(min_value)
remaining_seconds = int((min_value - whole_minutes) * 60)
# Add the minutes and seconds to the start_time to get the arrival time
arrive_time_dt = starting_date + timedelta(minutes=whole_minutes, seconds=remaining_seconds)
arrive_time_str = arrive_time_dt.isoformat()

def data_download():

    influxdb_org = "paros"
    influxdb_url = "https://influxdb.paros.casa.umass.edu/"
    # create influxdb client and API objects
    influxdb_client = InfluxDBClient(
        url=influxdb_url,
        token=influxdb_apikey,
        org=influxdb_org,
        verify_ssl=False
    )

    influxdb_query_api = influxdb_client.query_api()
    
    # Main Query
    
    idb_query = 'from(bucket:"' + "parosbox" + '")'\
        '|> range(start:' + start_time + 'Z, stop:' + end_time + 'Z)'\
        '|> filter(fn: (r) => r["_measurement"] == "' + box + '" )'\
        '|> filter(fn: (r) => r["id"] == "' + sensor_id + '")'

    
    device_result = influxdb_query_api.query(org=influxdb_org, query=idb_query)


    data_list = []
    sensor_time_list = []
    for table in device_result:
        for record in table.records:
            data_list.append(record.get_value())
            sensor_time_list.append(record.get_time())


    # Create a DataFrame for easy manipulation
    df = pd.DataFrame(data=data_list, index=sensor_time_list, columns=['value'])
    df = pd.DataFrame({
    'sensor_time': sensor_time_list, 
    'value': data_list})
    # Filter out rows with string values in the 'value' column
    df = df.loc[df['value'].apply(lambda x: isinstance(x, float))]

    # Convert 'value' column to numeric type
    df['value'] = pd.to_numeric(df['value'])
    df.to_csv(save_file_name, sep=',', index=True)
    #print(df)

    return df

## function of calculate the moving average (exponantial smoothing)
# Formula : https://en.wikipedia.org/wiki/Exponential_smoothing
def exp_smoothing_from_raw(df):
    # print(df)
    start_pressure = df['value'].iloc[0]  # pressure of the starting time.
    raw = []  # raw data list
    exp_smooth = []  # list for the exponential smoothing algorithm
    exp_smooth.append(start_pressure)
    raw.append(float(start_pressure))
    interval = int((1/sample_rate) * 60 * box_hz)

    time_points = [df.index[0]]  # list to store time points

    for i in range(1, len(df), interval):
        a = i
        b = int((i-1) / interval)
        x_t = df['value'].iloc[a]
        raw.append(float(x_t))
        predict = alpha * float(x_t) + (1 - alpha) * float(exp_smooth[b])
        exp_smooth.append(predict)
        time_points.append(df.index[a])

    # Creating a new DataFrame for the exponentially smoothed data
    smoothed_df = pd.DataFrame(data=exp_smooth, index=time_points, columns=['ESValue'])
    raw_df = pd.DataFrame(data=raw, index=time_points, columns=['RawValue'])
    # Calculating the residuals
    residual = [raw_val - smooth_val for raw_val, smooth_val in zip(raw, exp_smooth)]
    residual_df = pd.DataFrame(data=residual, index=time_points, columns=['Residual'])

    return raw_df, smoothed_df, residual_df

## function of ARIMA model
def arima_model(raw_df):

    model = ARIMA(raw_df['RawValue'], order=(p,d,q))
    results_ARIMA = model.fit()
    y3_series = pd.Series(results_ARIMA.predict())
    y3 = np.roll(y3_series, -1) # move left by one sample

    # Create a DataFrame for easy manipulation
    arima_df = pd.DataFrame(data=y3[:-1], index=raw_df.index[:-1], columns=['ARIMA'])

    return arima_df

def calculate_derivatives(df):

    # Calculate change in 'value'
    delta_value = df['RawValue'].diff()
    
    # Calculate change in 'time' (in seconds)
    delta_time = df.index.to_series().diff().dt.total_seconds()
    
    # Calculate first derivative (change in 'value' divided by change in 'time')
    first_derivative = delta_value / delta_time
    
    # Calculate second derivative
    second_derivative = first_derivative.diff() / delta_time[1:]
    
    # Create new DataFrames for the first and second derivatives
    first_derivative_df = pd.DataFrame(first_derivative.fillna(0), columns=['FirstDerivative'])
    second_derivative_df = pd.DataFrame(second_derivative.fillna(0), columns=['SecondDerivative'])

    return first_derivative_df, second_derivative_df

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
def fft_ac(df):
    data = df.values.flatten()

    # Calculate the number of windows
    n_windows = max(0, int((len(data) - 2**A) / 2**B) + 1)

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

    # Convert to dB if desired
    if dB:
        squared_sums = 10 * np.log10(squared_sums)

    # Create output DataFrame
    fft_df = pd.DataFrame(data=squared_sums, index=[df.index[i*2**B] for i in range(n_windows)], columns=['FFTValue'])

    # print(fft_df)

    return fft_df

## function of calculating the bandpass filtered (0.5Hz to 3Hz) pressure
def bpf_pressure(raw_df):

    pressure = raw_df['RawValue'].values

    n = len(pressure)  # number of data points
    timestep = 1 / (sample_rate/60)  # time between data points (in seconds)
    freq = np.fft.fftfreq(n, d=timestep)  # frequency array

    # apply FFT
    pressure_fft = np.fft.fft(pressure)

    # apply bandpass filter
    min_freq = 0.5  # Hz
    max_freq = 3  # Hz
    pressure_fft_filtered = pressure_fft.copy()  # create copy of FFT data
    pressure_fft_filtered[np.abs(freq) < min_freq] = 0  # set frequencies below min_freq to zero
    pressure_fft_filtered[np.abs(freq) > max_freq] = 0  # set frequencies above max_freq to zero

    # convert back to time domain
    pressure_filtered = np.fft.ifft(pressure_fft_filtered).real.tolist()  # convert back to list
    
    # Set the first 20 and last 20 values to zero using a loop
    for i in range(100):
        pressure_filtered[i] = 0  # Set the first 20 values to zero
        pressure_filtered[-(i + 1)] = 0  # Set the last 20 values to zero

    # Create new DataFrame with the same index
    bpf_raw_df = pd.DataFrame(data=pressure_filtered, index=raw_df.index, columns=['BPF_RawValue'])

    return bpf_raw_df

## function of plotting the raw data, exp smoothing, and ARIMA prediction for single barometer
def gen_main_graphs(raw, smooth, ARIMA):
    fig_main = plt.figure(figsize=(15, 10))
    st = "%s/%s/%s %s:%s"%(starting_date.year, starting_date.month, starting_date.day, starting_date.hour, starting_date.minute)
    fig_main.suptitle("%s\nRaw vs ARIMA prediction vs Exponential Smoothing \n (starting time is %s:00)"%(event_name, st), fontsize=18, fontweight = 'bold')
    fig_main.supxlabel("Timestamp (UTC)", fontsize=20,fontweight = 'bold')
    fig_main.supylabel("Pressure (hPa)", fontsize=20, fontweight = 'bold')
    title_2 = 'ARIMA (p,d,q = %s,%s,%s)'%(p,d,q)
    title_1 = 'E.S. (alpha = %s)'%(alpha)

    plt.plot(raw.index, raw['RawValue'], label ='raw - %s'%(device_name), linewidth=0.8, color = 'black')
    plt.plot(smooth.index, smooth['ESValue'], label = title_1, linewidth= 0.8, color = 'red')
    plt.plot(ARIMA.index, ARIMA['ARIMA'], label = title_2, linewidth=1, color = 'orange')
    # Draw a vertical line at chosen_time
    if timestamp_line == True:
        plt.axvline(x=arrive_time_dt, color='blue', linestyle='--')
        plt.text(0.8, 0.1, "the blue line indicates the time\nwhen the sound of the earthquake arrives", ha="center", va="center", transform=plt.gca().transAxes, fontsize = 12)


    plt.legend(fontsize=15, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    # plt.ylim(min(y2)-0.02, max(y2)+0.02)
    # plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.show()


    return ()

## function of plotting the residual (raw - ES), first derivative, second derivative for single barometer
def gen_diff_graphs(first_derivative, second_derivative, residual):
    
    fig = plt.figure(figsize=(15, 10))
    st = "%s/%s/%s %s:%s"%(starting_date.year, starting_date.month, starting_date.day, starting_date.hour, starting_date.minute)
    fig.suptitle("%s\nResiduals (raw - E.S.) and Derivative of Raw Data\n (starting time is %s:00)"%(event_name, st), fontsize=18, fontweight = 'bold')
    fig.supxlabel("Timestamp (UTC)", fontsize=20,fontweight = 'bold')
    fig.supylabel("Pressure (hPa)", fontsize=20, fontweight = 'bold')
    title_1 = 'First Derivative'
    title_2 = 'Second Derivative'
    title_3 = 'Residuals (raw - E.S.)'

    plt.plot(first_derivative.index, first_derivative['FirstDerivative'], label = title_1, linewidth=1, color = "blue")  # plot First Derivative
    # plt.plot(second_derivative.index, second_derivative['SecondDerivative'], label = title_2, linewidth=1, color = "green") # plot Second Derivative
    plt.plot(residual.index, residual['Residual'], label = title_3, linewidth=1, color = "orange") # plot Residuals (raw - E.S.)
    plt.text(0.1, 0.1, device_name, ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, facecolor='white', framealpha=1)
    # plt.ticklabel_format(useOffset=False, style='plain')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.xticks(fontsize=16)
    # plt.ticklabel_format(useOffset=False, style='plain')
    # formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 1000)))
    # plt.gca().xaxis.set_major_formatter(formatter)
    if timestamp_line == True:
        plt.axvline(x=arrive_time_dt, color='blue', linestyle='--')
        plt.text(0.8, 0.1, "the blue line indicates the time\nwhen the sound of the earthquake arrives", ha="center", va="center", transform=plt.gca().transAxes, fontsize = 12)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    plt.show()
    return()

# function of plotting the FFT of the residual(raw - ES) for single barometer
def gen_fft_ac_graph(fft):
    
    fft_window_size = 2**A
    fig_main = plt.figure(figsize=(15, 10))
    st = "%s/%s/%s %s:%s"%(starting_date.year, starting_date.month, starting_date.day, starting_date.hour, starting_date.minute)
    fig_main.suptitle("%s\nPower (magnitude of the sum of the FFT)\n(starting time is %s:00)"%(event_name, st), fontsize=18, fontweight = 'bold')
    fig_main.supxlabel("Timestamp (UTC)", fontsize=20,fontweight = 'bold')
    if dB == True:
        fig_main.supylabel("Power (Pa/Hz) in dB", fontsize=20, fontweight = 'bold')
    else:
        fig_main.supylabel("Power (Pa/Hz)", fontsize=20, fontweight = 'bold')
    title_1 = 'Sum of the FFT in the window of %s samples shift by %s'%(fft_window_size,2**B)


    plt.plot(fft.index, fft['FFTValue'], label = title_1, linewidth=1, color = 'magenta')
    plt.text(0.1, 0.1, device_name, ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    plt.legend(fontsize=12, bbox_to_anchor=[0.5, 1], loc='center', ncol=3, framealpha=1)
    if timestamp_line == True:
        plt.axvline(x=arrive_time_dt, color='blue', linestyle='--')
        plt.text(0.8, 0.1, "the blue line indicates the time\nwhen the sound of the earthquake arrives", ha="center", va="center", transform=plt.gca().transAxes, fontsize = 12)
    plt.xticks(fontsize=16)

    plt.show()

    return()

## function of plotting the bandpass filtered (0.5Hz to 3Hz) pressure
def gen_bpf_graph(bpf):
    
    fig_main = plt.figure(figsize=(15, 10))
    st = "%s/%s/%s %s:%s"%(starting_date.year, starting_date.month, starting_date.day, starting_date.hour, starting_date.minute)
    fig_main.suptitle("%s\nPressure (bandpass filtered to 0.5Hz-3Hz)\n(starting time is %s:00)"%( event_name, st), fontsize=18, fontweight = 'bold')
    fig_main.supxlabel("Timestamp (UTC)", fontsize=20,fontweight = 'bold')
    fig_main.supylabel("Pressure (hPa)", fontsize=20, fontweight = 'bold')


    plt.plot(bpf.index, bpf['BPF_RawValue'], linewidth=1, color = 'magenta')
    plt.text(0.1, 0.1, device_name, ha="center", va="center", transform=plt.gca().transAxes, fontsize = 18, fontweight = 'bold')
    if timestamp_line == True:
        plt.axvline(x=arrive_time_dt, color='blue', linestyle='--')
        plt.text(0.8, 0.1, "the blue line indicates the time\nwhen the sound of the earthquake arrives", ha="center", va="center", transform=plt.gca().transAxes, fontsize = 12)
    plt.xticks(fontsize=16)

    plt.show()

    return()

def main():

    origninal_raw_data = data_download()
    raw, exp, residual = exp_smoothing_from_raw(origninal_raw_data)
    Quit = False
    again = "y"


    while again == "y":
        while Quit == False:
            print("Which plot do you want to generate?\n     1. raw data, exp smoothing, and ARIMA prediction\n     2. residual(raw - ES), first derivative, second derivative\n     3. power fft\n     4. power fft with bandpass filter")
            plot_option = input("Enter your choice.(1, 2, 3 or 4): ")
            if plot_option == "1": # generate plot for raw data, exp smoothing, and ARIMA prediction for single paro
                arima = arima_model(raw)  
                gen_main_graphs(raw, exp, arima)
                break
            elif plot_option == "2": # generate plot for raw - ES, first derivative, second derivative for single paro
                first_derivative_df, second_derivative_df = calculate_derivatives(raw)
                gen_diff_graphs(first_derivative_df, second_derivative_df, residual)
                break
            elif plot_option == "3":  # generate power fft over time for single barometer
                fft_sum = fft_ac(residual)
                gen_fft_ac_graph(fft_sum)
                break
            elif plot_option == "4": # generate power fft over time with bandpass filter for single barometer
                bpf = bpf_pressure(raw)
                gen_bpf_graph(bpf)
            else:
                print("Invalid option. Please enter again")

        again = input("generate another plot(y/n): ")
        if again == "y":
            None
        elif again == "n":
            Quit == True
        else:
            print("Please enter again")
            again = input("generate another plot(y/n): ")

    return()

if __name__ == "__main__":
    main()#