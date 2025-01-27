"""
lfp_spike_window_analysis.py - Comprehensive module for LFP-spike analysis integration
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union
from neurodsp import spectral
from fooof import FOOOF

# Custom Exceptions -----------------------------------------------------------
class LFPSpikeWindowAnalysisError(Exception):
    """Base class for all LFP-spike window analysis errors"""
    pass

class DataMismatchError(LFPSpikeWindowAnalysisError):
    """Raised when input data dimensions don't match"""
    pass

class InvalidParameterError(LFPSpikeWindowAnalysisError):
    """Raised for invalid input parameters"""
    pass

class SpectralComputationError(LFPSpikeWindowAnalysisError):
    """Raised for errors in spectral computation"""
    pass

class FOOOFFitError(LFPSpikeWindowAnalysisError):
    """Raised when FOOOF model fitting fails"""
    pass

class WindowIndexError(LFPSpikeWindowAnalysisError):
    """Raised for invalid window indices"""
    pass

# LFP Analysis Core Functions --------------------------------------------------
from fooof import FOOOF
import numpy as np
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt
import warnings
from scipy.signal import welch

class LFPAnalysisError(Exception):
    """Base class for LFP analysis errors"""
    pass

class InvalidParameterError(LFPAnalysisError):
    """Raised for invalid input parameters"""
    pass

class SpectralComputationError(LFPAnalysisError):
    """Raised for errors in spectral computation"""
    pass

class FOOOFFitError(LFPAnalysisError):
    """Raised when FOOOF model fitting fails"""
    pass



def compute_lfp_windows(
    lfp_signal: np.ndarray,
    fs: float,
    window_length_sec: int = 25,
    step_size_sec: int = 15,
    freq_range: Tuple[float, float] = (5, 90),
    fooof_params: Dict = None,
    plot: bool = False
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[FOOOF]]:
    """
    Compute sliding windows of LFP data and extract spectral features using FOOOF.

    Args:
        lfp_signal (np.ndarray): Filtered LFP signal.
        fs (float): Sampling frequency in Hz.
        window_length_sec (int): Length of each sliding window in seconds.
        step_size_sec (int): Step size between windows in seconds.
        freq_range (Tuple[float, float]): Frequency range for FOOOF fitting.
        fooof_params (Dict): Parameters for the FOOOF model.
        plot (bool): Whether to plot power spectra for each window.

    Returns:
        Tuple[List[Tuple[np.ndarray, np.ndarray]], List[FOOOF]]:
            - spectra: List of tuples containing frequency and power spectra.
            - foof_results: List of FOOOF objects with fitted features.
    """
    # Convert window and step size to samples
    window_length = int(window_length_sec * fs) 
    step_size = int(step_size_sec * fs)

    # Initialize results storage
    spectra = []
    foof_results = []

    # Define window start and end times
  
    # Define window times in samples
    window_times = [
    (start, min(start + window_length, len(lfp_signal)))
    for start in range(0, len(lfp_signal), step_size)
]




    # Loop through sliding windows
    for start, end in window_times:
        # Extract the LFP segment
        lfp_segment = lfp_signal[start:end]
        
        # Compute the power spectrum using Welch's method
        try:
            fxx, pxx = spectral.compute_spectrum(
                lfp_segment, fs, method='welch', window='hann', nperseg=fs * 4
            )
        except ValueError as e:
            raise SpectralComputationError(f"Error computing power spectrum: {str(e)}") from e
        
        # Fit the FOOOF model
        fm = FOOOF(**(fooof_params or {"max_n_peaks": 4, "verbose": False}))
        try:
            fm.fit(fxx, pxx, freq_range=freq_range)
        except Exception as e:
            raise FOOOFFitError(f"FOOOF fitting failed for window {start}-{end}: {str(e)}") from e

        # Store results
        spectra.append((fxx, pxx))
        foof_results.append(fm)

        # Optional: plot the power spectrum
        if plot:
            plt.figure(figsize=(8, 6))
            plt.loglog(fxx, pxx, label="Power Spectrum")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power (V^2/Hz)")
            plt.title(f"LFP PSD: {start / fs:.2f}s - {end / fs:.2f}s")
            plt.legend()
            plt.show()

    return spectra, foof_results



# Spike-LFP Integration Functions ----------------------------------------------
def map_spikes_to_windows(
    spk_times_ms: List[float],
    spk_ids: List[int],
    window_times: List[Tuple[int, int]],
    df_spike_ids: Union[pd.Series, List[int]],
    fs: float
) -> Dict[int, List[int]]:
    """
    Map spike IDs to LFP windows with sample-to-ms conversion
    
    Args:
        spk_times_ms: Spike timestamps in milliseconds
        spk_ids: Unique spike identifiers
        window_times: List of (start, end) sample indices
        df_spike_ids: Valid spike IDs from DataFrame
        fs: Sampling frequency for conversion
        
    Returns:
        Dictionary mapping spike IDs to window indices
    """
    # Convert window times to milliseconds
    window_times_ms = [
        (start/fs*1000, end/fs*1000) 
        for (start, end) in window_times
    ]


    return _map_spikes_to_window_helper(
        spk_times_ms, 
        spk_ids, 
        window_times_ms, 
        df_spike_ids
    )

def _map_spikes_to_window_helper(
    spk_times_ms: List[float],
    spk_ids: List[int],
    window_times_ms: List[Tuple[float, float]],
    df_spike_ids: Union[pd.Series, List[int]]
) -> Dict[int, List[int]]:
    """Core mapping logic with milliseconds-based windows"""
    # Input validation
    if len(spk_times_ms) != len(spk_ids):
        raise DataMismatchError(
            f"spk_times_ms ({len(spk_times_ms)}) and spk_ids ({len(spk_ids)}) must match"
        )

    valid_spike_ids = set(df_spike_ids) if isinstance(df_spike_ids, pd.Series) else set(df_spike_ids)
    spike_to_window_map = {}


    for spk_id, spike_ms in zip(spk_ids, spk_times_ms):
        if spk_id not in valid_spike_ids:
            warnings.warn(f"Spike ID {spk_id} not in DataFrame - skipping", UserWarning)
            continue
            
        spike_to_window_map[spk_id] = []
        for window_idx, (start, end) in enumerate(window_times_ms):
            if start <= spike_ms <= end + 1:
                spike_to_window_map[spk_id].append(window_idx)

    return spike_to_window_map

def combine_spike_lfp_features(
    spike_data: pd.DataFrame,
    foof_results: List[FOOOF],
    spike_to_window_map: Dict[int, List[int]],
    lfp_prefix: str = "lfp_"
) -> pd.DataFrame:
    """
    Merge FOOOF features into spike DataFrame
    
    Args:
        spike_data: DataFrame with spike parameters
        foof_results: List of FOOOF objects from create_lfp_windows
        spike_to_window_map: Mapping from map_spikes_to_windows
        lfp_prefix: Prefix for LFP feature columns
        
    Returns:
        Updated DataFrame with LFP features
    """
    # Input validation
    if 'spk_id' not in spike_data.columns:
        raise MissingColumnError("DataFrame must contain 'spk_id' column")

    df = spike_data.copy()
    lfp_features = [
        'offset_current', 'exponent_current', 'r_squared_current',
        'error_current', 'n_peaks_current',
        'offset_previous', 'exponent_previous',
        'r_squared_previous', 'error_previous', 'n_peaks_previous'
    ]
    
    for feat in lfp_features:
        df[f"{lfp_prefix}{feat}"] = np.nan

    for spk_id, window_indices in spike_to_window_map.items():
        row = df[df['spk_id'] == spk_id]
        if row.empty:
            continue
            
        row_idx = row.index[0]
        
        # Current window processing
        if window_indices:
            current_idx = window_indices[0]
            if 0 <= current_idx < len(foof_results):
                _add_fooof_features(df, row_idx, foof_results[current_idx], f"{lfp_prefix}current")
        
        # Previous window processing
        if window_indices and window_indices[0] > 0:
            prev_idx = window_indices[0] - 1
            if 0 <= prev_idx < len(foof_results):
                _add_fooof_features(df, row_idx, foof_results[prev_idx], f"{lfp_prefix}previous")

    return df

def _add_fooof_features(
    df: pd.DataFrame,
    row_idx: int,
    foof_obj: FOOOF,
    prefix: str
) -> None:
    """Helper to add FOOOF features to DataFrame row"""
    try:
        df.at[row_idx, f"{prefix}_offset"] = foof_obj.aperiodic_params_[0]
        df.at[row_idx, f"{prefix}_exponent"] = foof_obj.aperiodic_params_[-1]
        df.at[row_idx, f"{prefix}_r_squared"] = foof_obj.r_squared_
        df.at[row_idx, f"{prefix}_error"] = foof_obj.error_
        df.at[row_idx, f"{prefix}_n_peaks"] = foof_obj.n_peaks_
    except AttributeError as e:
        raise FOOOFFitError(f"Missing FOOOF feature: {str(e)}") from e