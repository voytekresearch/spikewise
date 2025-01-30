# signal_utils.py

import numpy as np
from scipy.signal import sosfiltfilt, butter

def butter_bandpass(data: np.ndarray, fs: float, filt_freq: list, order: int = 4) -> np.ndarray:
    """Original sequential highpass + lowpass filter from your code."""
    if len(filt_freq) != 2:
        raise ValueError("filt_freq must contain exactly two frequencies")
    
    nyq = 0.5 * fs
    # Highpass first
    sos_high = butter(order, filt_freq[0]/nyq, btype='high', output='sos')
    y = sosfiltfilt(sos_high, data)
    # Then lowpass
    sos_low = butter(order, filt_freq[1]/nyq, btype='low', output='sos')
    return sosfiltfilt(sos_low, y)