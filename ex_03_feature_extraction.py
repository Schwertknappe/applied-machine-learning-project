import numpy as np
import pandas as pd
from scipy.signal import detrend, windows, find_peaks


def find_dominant_frequencies(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Calculates the dominant frequencies of multiple input signals with the fast fourier transformation.

    Args:
        x (np.ndarray): The input signals, shape: (num_samples, seq_len).
        fs (int): The sampling frequency of the signals.

    Returns:
        np.ndarray: The dominant frequencies for each signal, shape: (num_samples,).
    """
    # initialize empty array with shape matching the first dimension of the input array x
    dominant_frequencies = np.zeros(x.shape[0])

    # loop through each signal to find the dominant frequency
    for i, signal in enumerate(x):
        # Detrending
        #   remove linear trends affecting all data points -> center signal around mean
        detrended_signal = detrend(signal)

        # Windowing
        #   using Hann windowing as nature/shape of signal is unknown/likely combination of sines (?)
        #   apply window by multiplying element-wise with the detrended signal
        window = windows.hann(len(detrended_signal))
        windowed_signal = detrended_signal * window

        # Compute fast fourier transform (FFT)
        #   converts signal from time-domain to frequence-domain
        #   get magnitude to simplify values from complex to real and discard mirrored negative values
        fft_result = np.fft.fft(windowed_signal)
        fft_magnitude = np.abs(fft_result)
        fft_magnitude = fft_magnitude[:len(fft_magnitude)//2]

        # Compute PSD
        #   PSD helps in separating dominant frequencies from noise
        #   take the squared absolute fft values to the power of 2 and divide by sampling frequency * sequence length
        psd = (fft_magnitude ** 2) / (fs * len(windowed_signal))

        # Frequency axis
        #   create Hz axis of frequencies aligned with PSD result axis
        frequencies = np.fft.fftfreq(len(windowed_signal), 1/fs)[:len(windowed_signal)//2]

        # Find the dominant frequency
        #   get maximum value in calculated PSD and store it in appropriate place in return array
        dominant_freq = frequencies[np.argmax(psd)]
        dominant_frequencies[i] = dominant_freq

    return dominant_frequencies


def extract_features(data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Extract 20 different features from the data.
    Args:
        data (np.ndarray): The data to extract features from.
        labels (np.ndarray): The labels of the data.
    Returns:
        pd.DataFrame: The extracted features.
    """
    fs = 100_000

    # Separate voltage and current signals
    currents = data[:, :, 0]
    voltages = data[:, :, 1]

    # Frequency-domain features using existing function
    curr_dom_freqs = find_dominant_frequencies(currents, fs)
    volt_dom_freqs = find_dominant_frequencies(voltages, fs)

    # setup empty array to fill in for each given sample
    features = []

    for i in range(data.shape[0]):
        volt = voltages[i]
        curr = currents[i]
        power = volt * curr

        volt_peaks, _ = find_peaks(volt)
        curr_peaks, _ = find_peaks(curr)

        volt_time_to_peak = np.argmax(volt) / fs
        curr_time_to_peak = np.argmax(curr) / fs

        feat = {
            # Voltage features
            'volt_mean': np.mean(volt),
            'volt_std': np.std(volt),
            'volt_min': np.min(volt),
            'volt_max': np.max(volt),
            'volt_median' : np.median(volt),
            'volt_num_peaks': len(volt_peaks),
            'volt_time_to_peak': volt_time_to_peak,

            # Current features
            'curr_mean': np.mean(curr),
            'curr_std': np.std(curr),
            'curr_min': np.min(curr),
            'curr_max': np.max(curr),
            'curr_median': np.median(curr),
            'curr_num_peaks': len(curr_peaks),
            'curr_time_to_peak': curr_time_to_peak,

            # Power features
            'power_mean': np.mean(power),
            'power_std': np.max(power),
            'power_min': np.min(power),
            'power_max': np.max(power),

            # Frequency domain
            'volt_dom_freq': volt_dom_freqs[i],
            'curr_dom_freq': curr_dom_freqs[i],
        }

        features.append(feat)

    df = pd.DataFrame(features)
    df['labels'] = labels

    return df

