import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from io import StringIO


# Read the data into a DataFrame
df = pd.read_csv('e15t2.txt')

# Convert the timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['value'])

# Calculate the sampling frequency

sampling_freq = 20
# Design a high-pass filter with a cutoff frequency of 1.5 Hz
cutoff_freq = 1.5  # Cutoff frequency in Hz
nyquist_freq = 0.5 * sampling_freq
normal_cutoff = cutoff_freq / nyquist_freq
b, a = butter(1, normal_cutoff, btype='high', analog=False)

# Apply the filter
filtered_signal = filtfilt(b, a, df['value'].values)

# Add the filtered signal to the DataFrame
df['filtered_signal'] = filtered_signal

# Save the DataFrame to a new text file, excluding the intermediate columns
df[['sensor_time', 'filtered_signal']].to_csv('filtered_e15t2.txt', index=False)