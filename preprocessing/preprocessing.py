import copy

import wfdb
import numpy as np
from math import sqrt, pi
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy.signal import find_peaks
from scipy.signal import resample
from new_filter import detect_peaks
from scipy.stats import pearsonr


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def save_sample(array, file_path):
    normalized_array = (array - array.min()) / (array.max() - array.min())
    np.save(file_path, normalized_array)
    return file_path


import numpy as np
from scipy.signal import butter, filtfilt
import os

errs = []

# directory = "git_ecg_data_full"

# Extension to look for

# Iterate through all files in the directory
# for filename in os.listdir(directory):
#     # Check if the file ends with the .dat extension
#     if filename.endswith(extension):
#         # Remove the extension and print the filename
#         file_without_extension = os.path.splitext(filename)[0]
#         print(file_without_extension)


import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some inputs.')

    # Add arguments
    parser.add_argument('--test', type=bool, default=True, help='A boolean to set the test mode')
    parser.add_argument('--raw_ecg_dir', type=str, default="ec_data_full", help='Directory for raw ECG data')
    parser.add_argument('--sample_ecg_dir', type=str, default="ec_samples", help='Directory for ECG samples')
    parser.add_argument('--sample_ecg_test_dir', type=str, default="ec_test_samples",
                        help='Directory for ECG test samples')

    args = parser.parse_args()
    TEST = True

    # raw_ecg_dir = "git_ecg_data_full"
    # sample_ecg_dir = "git_ecg_samples"
    # sample_ecg_test_dir = "git_ecg_test_samples"


    i = 0
    extension = ".dat"

    TEST = args.test
    raw_ecg_dir = args.raw_ecg_dir
    sample_ecg_dir = args.sample_ecg_dir
    sample_ecg_test_dir = args.sample_ecg_test_dir


    os.makedirs(sample_ecg_dir, exist_ok=True)
    os.makedirs(sample_ecg_test_dir, exist_ok=True)

    print(f"Test mode: {TEST}")
    print(f"Raw ECG data directory: {raw_ecg_dir}")
    print(f"Sample ECG directory: {sample_ecg_dir}")
    print(f"Sample ECG test directory: {sample_ecg_test_dir}")

    for filename in os.listdir(raw_ecg_dir):
        # Check if the file ends with the .dat extension
        if filename.endswith(extension):
            i+=1
            # Remove the extension and print the filename
            file_without_extension = os.path.splitext(filename)[0]
            print(file_without_extension)
            file_num = int(file_without_extension)
            record = wfdb.rdrecord(f'{raw_ecg_dir}/{file_without_extension}')
            # if file_without_extension != "0110":
            #     continue


            signal = record.p_signal
            sampling_rate = record.fs



            # CHANGES HERE
            if TEST:
                signal = signal[300000:400000, 0]
            else:
                signal = signal[:300000, 0]

            # if file_without_extension=="0110":
            #     plt.plot(signal)
            #     plt.show()

            # FILTERING START

            orig_sig = copy.deepcopy(signal)

            # FILTER 1
            # Filter requirements.
            order = 1
            fs = sampling_rate
            cutoff = 0.2

            y = butter_lowpass_filter(signal, cutoff, fs, order)
            T = len(signal) / fs
            n = int(T * fs)  # total number of samples
            t = np.linspace(0, T, n, endpoint=False)
            signal = signal - (np.concatenate((y[500:], np.repeat(y[-1], 500))))
            # FILTER 1 END

            # FILTER 2
            order = 1
            fs = sampling_rate
            cutoff = 0.3

            y = butter_lowpass_filter(signal, cutoff, fs, order)
            T = len(signal) / fs
            n = int(T * fs)  # total number of samples
            t = np.linspace(0, T, n, endpoint=False)
            signal = signal - (np.concatenate((y[500:], np.repeat(y[-1], 500))))

            # FILTER 2 END

            # High frequency noise reduction

            # FILTER 3
            order = 4
            fs = sampling_rate
            cutoff = 25

            y = butter_lowpass_filter(signal, cutoff, fs, order)
            T = len(signal) / fs
            n = int(T * fs)
            t = np.linspace(0, T, n, endpoint=False)

            normalized_array = (y - np.min(y)) / (np.max(y) - np.min(y))
            filtered_signal = normalized_array

            # FILTER 3 END

            # FILTERING END

            peaks = detect_peaks(filtered_signal, fs, to_plot=False)
            print(len(peaks))

            # Data sampling
            LEFT_BOUND = 250
            RIGHT_BOUND = 450
            print(f"##################\n  {i}\n###############")
            print(f"peaks len {len(peaks)}")
            if len(peaks) < 50:
                print(f"\n\n\n\n\n{i}th data is corrupted\n\n\n\n")
                continue
            one_sample = filtered_signal[peaks[1] - LEFT_BOUND:peaks[1] + RIGHT_BOUND]

            if TEST:
                SAMPLE_NUM = 31
            else:
                SAMPLE_NUM = 201
            mean_sample = np.zeros(300)


            # for j in range (1,len(peaks)-1):
            for j in range(1, SAMPLE_NUM):
                # RESAMPLING TO 300 POINTS (428 Hz)
                resampled_signal = resample(filtered_signal[peaks[j] - LEFT_BOUND:peaks[j] + RIGHT_BOUND], 300)
                if TEST:

                    dir_path = f"{sample_ecg_test_dir}/{file_num}"
                    os.makedirs(dir_path, exist_ok=True)
                    save_sample(resampled_signal, f"{sample_ecg_test_dir}/{file_num}/subject{i}_sample{j}.npy")
                else:

                    dir_path = f"{sample_ecg_dir}/{file_num}"
                    os.makedirs(dir_path, exist_ok=True)
                    save_sample(resampled_signal, f"{sample_ecg_dir}/{file_num}/subject{i}_sample{j}.npy")


