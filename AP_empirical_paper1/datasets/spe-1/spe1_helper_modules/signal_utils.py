# signal_utils.py

import numpy as np
from scipy.signal import sosfiltfilt, butter


def butter_bandpass(data: np.ndarray, fs: float, filt_freq: list, order: int = 4) -> np.ndarray:
    """
    Apply a bandpass Butterworth filter to a signal using second-order sections (SOS).
    
    Parameters:
        data (np.ndarray): The input signal.
        fs (float): Sampling rate in Hz.
        filt_freq (list or tuple): Two-element list or tuple with [low, high] cutoff frequencies in Hz.
        order (int): Filter order (recommended: 4 or higher).
    
    Returns:
        np.ndarray: The filtered signal.
    """
    if not isinstance(filt_freq, (list, tuple)) or len(filt_freq) != 2:
        raise ValueError("filt_freq must be a list or tuple of two numbers [low, high]")

    nyq = 0.5 * fs
    low = filt_freq[0] / nyq
    high = filt_freq[1] / nyq

    if low <= 0 or high >= 1 or low >= high:
        raise ValueError("Cutoff frequencies must be within (0, Nyquist) and low < high")

    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)


def butter_highpass(data: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase highpass Butterworth filter.

    Parameters
    ----------
    data   : 1-D signal
    fs     : sampling rate (Hz)
    cutoff : highpass cutoff frequency (Hz)
    order  : filter order (default 4)
    """
    nyq = 0.5 * fs
    sos = butter(order, cutoff / nyq, btype='high', output='sos')
    return sosfiltfilt(sos, data)
