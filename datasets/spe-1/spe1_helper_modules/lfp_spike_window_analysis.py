"""
lfp_spike_window_analysis.py - Comprehensive module for LFP-spike analysis integration
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union
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
def create_lfp_windows(
    lfp_signal: np.ndarray,
    fs: float,
    window_length: float = 25.0,
    step_size: float = 15.0,
    spectral_params: dict = None,
    fooof_params: dict = None,
    plot: bool = False
) -> Tuple[List[Tuple[int, int]], List[tuple], List[FOOOF]]:
    """
    Create sliding windows and perform spectral analysis on LFP data.

    Args:
        lfp_signal: Raw LFP signal (1D array)
        fs: Sampling frequency (Hz)
        window_length: Window length in seconds
        step_size: Step size between windows in seconds
        spectral_params: Parameters for spectral computation
        fooof_params: Parameters for FOOOF model
        plot: Whether to plot power spectra

    Returns:
        Tuple containing:
        - window_times: List of (start, end) sample indices
        - spectra: List of (freqs, powers) tuples
        - foof_results: List of FOOOF objects

    Raises:
        InvalidParameterError: For invalid input parameters
        SpectralComputationError: If spectral computation fails
        FOOOFFitError: If FOOOF fitting fails
    """
    # Input validation
    if len(lfp_signal) == 0:
        raise InvalidParameterError("LFP signal cannot be empty")
    
    if fs <= 0:
        raise InvalidParameterError(f"Invalid sampling frequency: {fs} Hz")

    if window_length <= 0 or step_size <= 0:
        raise InvalidParameterError("Window length and step size must be > 0")

    # Convert time parameters to samples
    window_samples = int(fs * window_length)
    step_samples = int(fs * step_size)

    if window_samples > len(lfp_signal):
        raise InvalidParameterError(
            f"Window length ({window_length}s) exceeds signal duration "
            f"({len(lfp_signal)/fs:.2f}s)"
        )

    # Set default parameters
    default_spectral_params = {
        'method': 'welch',
        'window': 'hann',
        'nperseg': int(fs * 4)  # 4-second segments
    }
    spectral_params = {**default_spectral_params, **(spectral_params or {})}

    default_fooof_params = {
        'max_n_peaks': 4,
        'freq_range': [5, 90],
        'verbose': False
    }
    fooof_params = {**default_fooof_params, **(fooof_params or {})}

    # Create windows
    window_times = []
    spectra = []
    foof_results = []
    
    try:
        for start in range(0, len(lfp_signal) - window_samples, step_samples):
            end = start + window_samples
            window_times.append((start, end))

            # Compute power spectrum
            lfp_segment = lfp_signal[start:end]
            fxx, pxx = _compute_spectrum(
                lfp_segment, fs, 
                method=spectral_params['method'],
                window=spectral_params['window'],
                nperseg=spectral_params['nperseg']
            )

            # Fit FOOOF model
            fm = FOOOF(**fooof_params)
            if not fm.fit(fxx, pxx, fooof_params['freq_range']):
                raise FOOOFFitError(f"FOOOF fit failed for window {start}-{end}")
                
            spectra.append((fxx, pxx))
            foof_results.append(fm)

            if plot:
                plt.loglog(fxx, pxx)
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power (V^2/Hz)')
                plt.title(f"LFP PSD {start/fs:.2f}s-{end/fs:.2f}s")
                plt.show()

    except Exception as e:
        error_msg = f"Error processing window {start}-{end}: {str(e)}"
        if isinstance(e, FOOOFFitError):
            raise FOOOFFitError(error_msg) from e
        raise SpectralComputationError(error_msg) from e

    return window_times, spectra, foof_results

def _compute_spectrum(
    signal: np.ndarray,
    fs: float,
    method: str = 'welch',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal function for power spectrum computation
    
    Args:
        signal: Input signal
        fs: Sampling frequency
        method: Spectral method ('welch', 'multitaper', etc.)
        kwargs: Method-specific parameters
        
    Returns:
        (frequencies, power) tuple
        
    Raises:
        SpectralComputationError: If computation fails
    """
    try:
        from scipy.signal import welch
        fxx, pxx = welch(signal, fs=fs, **kwargs)
        return fxx, pxx
        
    except Exception as e:
        raise SpectralComputationError(
            f"Spectral computation failed ({method}): {str(e)}"
        ) from e

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
            if start <= spike_ms < end:
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