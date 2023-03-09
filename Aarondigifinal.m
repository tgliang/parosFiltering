%Infrasounddetect
%Authors: David Pepyne and Aaron Annan
%This code is used to filter the raw data provided by the Paroscientific
%6000-16B microbarometers and create spectrograms to display an infrasound
%signal of interest

%Paroscientific Barometer 6000-16B data analyzer
%input is a .txt file from the logging function of the Digiquartz
%Interactive 2.0 software installed on a Windows prior to use

clear all
close all
clc

%==========================================================================
% Section 1: Load Data File, Store time and pressure data into vectors,
% define number of samples
%==========================================================================


%Upon executing this script from the MATLAB command window, the user will
%be presented with the computer's file directory, from which the user will
%select the .txt file to be analyzed

[dat_filename, dat_dirname] = uigetfile('*.txt',...
    'Select a barometer data file');

%Define the number of sensors (barometers) used in test. This script is
%limited to analyzing the .txt file from one sensor.
numSensors = 1;

% Import the user-selected file into the memory of MATLAB's working
% directory
disp(['INFO - analyzing barometer data file : ', dat_dirname, '/', dat_filename]);
DAT = importdata(fullfile(dat_dirname, dat_filename));
disp(dat_filename);
disp(DAT);

%Store the pressure and time vectors of the .txt file into a matrix
%variable. Since the file is headed for the first 5 lines, the CSV's in the 
%numerical vectors begin at the 6th row, hence the "6,1".
filetxt = csvread([dat_filename],6,1); 

%Store the time vector (first column) of the txt file into a variable "t" 
%and the pressure vector (second column) into a variable "P". Define the
%number of pressure samples as the length of vector "P".
t = filetxt(:,1);
P = filetxt (:,2);
numSamples = length(P);

%==========================================================================
%==========================================================================

%==========================================================================
% Section 2: Plot the raw data (pressure vs. time)
%==========================================================================

%The raw pressure plot will emulate what is seen on the Digiquartz
%Interactive 2.0 software.
figure
% disp(numsamples)
plot(t, P, 'g');
titleone = 'Raw Pressure vs Time';
title(titleone);
xlabel('Time (s)');
ylabel('Pressure (hPa)');
%saveas(figureraw,fullfile('C:\Research Summer 2014 David McLaughlin\Barometer test data and Matlab analysis\Test 3\Matlab analysis', titleone), 'bmp');

%==========================================================================
%==========================================================================

%==========================================================================
% Section 3: Plot the mean-removed data
%==========================================================================

%Convert data units into Pa if not already done so. This for loop goes
%through
for i = 1:numSamples
    if P(i) < 10000
        P(i) = (100 * P(i));
    else
        ;
    end
end

figure
%P = 100*P;
mP = mean(P);
P1 = P - mP;
% detrendpressure = detrend(pressure); same functionality except uses
% MATLAB's built in detrend function
plot(t, P1, 'g');
titletwo = 'Mean-removed pressure vs. Time';
title(titletwo);
xlabel('Time (s)');
ylabel('Mean-removed Pressure (Pa)');
%saveas(figureraw,fullfile('C:\Research Summer 2014 David McLaughlin\Barometer test data and Matlab analysis\Test 3\Matlab analysis', titleone), 'bmp');

%==========================================================================
%==========================================================================

%==========================================================================
% Section 4: Create a Bandpass filter, this filters out the low frequency
% trends and high frequency roll-off effects, the low frequency variation
% can dominate the spectrum compared to the infrasound signal of interest.
% The script estimates the sampling frequency as the reciprocal of the
% mean of the interval between sampling times. The filter is a fourth-order
% butterworth.
%==========================================================================

%Estimate the sampling rate by first calculating the mean interval between 
%sampling times then converting to a frequency in the variable "samplerate" 

interval = diff(t); 
%creates a vector of intervals between the sample time values in the "time"
%vector
%disp(interval)
%disp(length(interval))
%disp(length(time))

ts = mean(interval); 
%stores the mean of the interval values in the 'interval' vector

Fs = 1 / ts;
%disp(samplerate)

filterorder = 4; %I do not know why the filter order is 4
lowcutoff = 0.25;
highcutoff = 0.75*Fs/2; %Frequencies of interest lie in between 0.5 and 4 Hz
lownormfreq = lowcutoff / (0.5 * Fs); 
highnormfreq = highcutoff / (0.5 * Fs);
[b,a] = butter(filterorder, [lownormfreq highnormfreq], 'bandpass');
% disp(a)
% disp(b)

% Use the band pass (filtfilt function) to obtain zero phase distortion and
% filter out start up transients
P2 = filtfilt(b,a,P1);
%P2
%P2(1:128)
%length(P2(385:512))

%==========================================================================
%==========================================================================

%==========================================================================
% Section 5: Plot the mean-removed-filtered data 
%==========================================================================

figurefilter = figure;
%figure
plot(t,P2,'g');
xlabel('Time (sec)');
ylabel('Amplitude (Pa)');
titlethree ='Mean removed-filtered barometric pressure';
title(titlethree);

%==========================================================================
%==========================================================================

tStart = input('Enter the analysis start time: ');
if ( isempty(tStart) ),
    tStart = -inf;
end    
I = find(t < tStart);
P2(I) = [];
t(I) = [];

tStop = input('Enter the analysis stop time: ');
if ( isempty(tStop) ),
    tStop = +inf;
end    
I = find(t > tStop);
P2(I) = [];
t(I) = [];

% update the total number of samples
numSamples = length(P2);

% plot the analysis segment
% figure
% plot(t,P2)


%==========================================================================
% Section 5: Create a Spectrogram of the data
%==========================================================================

% set N to the total number of samples
N = numSamples;

% define the desired block size, blocks will be used to average the smaples
% within the blocks
B = 1000;

% define the number of fft coefficients to generate
NFFT = 2^nextpow2(2*B-1);
%NFFT

% define the number of samples overlap between blocks
O = B/2;

% calculate the total number of blocks
M = (N - O) / (B - O);
M = fix(M);
%M

% define the starting and ending sample indices for each block
k1 = 1:(B-O):M*(B-O);
k2 = k1 + B - 1;
%k1(M)
%k2(M)

% Here, in order to mitigate spectral leakage and enhance the infrasound
% spectral bands, define the Hamming window function
w = hamming(B);
U = sum( w .* w ) / B;


% Designate a secion of memory for the predetermined set of  periodograms
% that will make up the spectrogram. Each periodogram will have a value of 
% coefficients equivalent to the variable NFFT, there will be be M
% periodograms 
PxxArray = zeros(NFFT/2+1,M);

% generate the frequency vector associated with the one-sided spectrum
df  = Fs/NFFT;               % frequency resolution
FV2 = (0:NFFT-1)*df;         % two-sided frequency vector
FV1 = FV2(1:fix(NFFT/2)+1);  % one-sided frequency vector

% generate the M periodograms
for i = 1:M,
    
%     Pxx= pwelch(x2(1,k1(i):k2(i)),[],[],NFFT,Fs);    

    % window the i-th timeseries block - here we only look at sensor 1
    wx = w .* P2(k1(i):k2(i));

    % take its fft
    X = fft(wx,NFFT);

    % calculate the peridogram for the i-th block
    Pxx = X .* conj(X);
    Pxx = Pxx / (Fs * B * U );  % note scaling by B
    
    % extract the one-sided spectrum from the two-sided periodogram - this
    % involves extracting the positive frequencies and doubling all
    % coefficients except DC and Nyquist
    Pxx1 = Pxx(1:fix(NFFT/2)+1);
    Pxx1(2:end-1) = 2*Pxx1(2:end-1);
    
    % store the one-sided spectrum in the spectrogram array
    PxxArray(:,i) = Pxx1;
    
    % plot the one-sided spectrum
    figure(1000)
    plot(FV1,20*log10(abs(Pxx1)))
    
%     disp('Press any key to continue...')
    pause(0.1)
    
end

% form a time vector for the spectrogram
TV = t(k1);

%Create the color-coded spectral plot
titleStr = sprintf('%s\nSensor %d Spectrogram (Sample Rate = %0.2f Hz, Frequency Resolution = %0.4f)',...
    dat_filename,'',Fs,df);
figure
surf(TV,FV1,20*log10(abs(PxxArray)))
title(titleStr)
xlabel('t (sec)')
ylabel('Frequency (Hz)')
shading flat
view(0,90)




