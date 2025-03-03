"""
lfp_spike_window_analysis.py - Comprehensive module for LFP-spike analysis integration
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union, Literal
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



import warnings

def compute_lfp_windows(
    lfp_signal: np.ndarray,
    fs: float,
    window_length_sec: int = 25,
    step_size_sec: int = 15,
    freq_range: Tuple[float, float] = (1, 90),
    fooof_params: Dict = None,
    plot: bool = False,
    min_overlap_percent: float = 50.0  # New parameter
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[FOOOF], List[Tuple[int, int]]]:
    """
    Compute sliding windows of LFP data with overlap validation.
    
    New Args:
        min_overlap_percent: Minimum required overlap percentage (0-100) to trigger warning
    """
    # Calculate actual overlap percentage
    actual_overlap = window_length_sec - step_size_sec
    overlap_percent = (actual_overlap / window_length_sec) * 100
    
    # Check for insufficient overlap
    if overlap_percent < min_overlap_percent:
        warnings.warn(
            f"\n\n⚠️ Insufficient window overlap: {overlap_percent:.1f}% "
            f"(minimum recommended: {min_overlap_percent}%)\n"
            "This may result in:\n"
            "1. Spikes mapping to only one window\n"
            "2. Gaps in temporal coverage\n"
            "3. Reduced statistical power\n\n"
            "Recommended fix:\n"
            f"Set step_size_sec <= {window_length_sec * (1 - min_overlap_percent/100):.1f} "
            f"for {min_overlap_percent}% overlap\n",
            UserWarning
        )
    
    # Convert window and step size to samples
    window_length = int(window_length_sec * fs)
    step_size = int(step_size_sec * fs)

   

    # Initialize results storage
    spectra = []
    foof_results = []
  
    # Define window start and end times
    window_times = [
        (start, start + window_length)
        for start in range(0, len(lfp_signal) - window_length + 1, step_size)
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
            # Optional: plot the power spectrum
            if plot:
                fm.plot(plot_peaks='shade', peak_kwargs={'color' : 'green'})
        

        except Exception as e:
            raise FOOOFFitError(f"FOOOF fitting failed for window {start}-{end}: {str(e)}") from e

        # Store results
        spectra.append((fxx, pxx))
        foof_results.append(fm)
      

    return spectra, foof_results, window_times


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
        (start / fs * 1000, end / fs * 1000) 
        for (start, end) in window_times
    ]

    # Call the helper function for mapping
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
        raise ValueError(
            f"spk_times_ms ({len(spk_times_ms)}) and spk_ids ({len(spk_ids)}) must match"
        )

    # Ensure the list of valid spike IDs is a set for fast lookup
    valid_spike_ids = set(df_spike_ids) if isinstance(df_spike_ids, pd.Series) else set(df_spike_ids)
    spike_to_window_map = {}



    for spk_id, spike_ms in zip(spk_ids, spk_times_ms):
        if spk_id not in valid_spike_ids:
            warnings.warn(f"Spike ID {spk_id} not in DataFrame - skipping", UserWarning)
            continue

        spike_to_window_map[spk_id] = []

        # Map spike to all overlapping windows
        for window_idx, (start, end) in enumerate(window_times_ms):
            # Spike falls completely within this window
            if start <= spike_ms < end:
                spike_to_window_map[spk_id].append(window_idx)
            # Spike is exactly on the boundary of two windows
            elif window_idx < len(window_times_ms) - 1 and spike_ms == end:
                spike_to_window_map[spk_id].append(window_idx + 1)

        

    return spike_to_window_map




def combine_spike_lfp_features(
    spike_data: pd.DataFrame,
    foof_results: List[FOOOF],
    spike_to_window_map: Dict[int, List[int]],
    lfp_prefix: str = "lfp_"
) -> pd.DataFrame:
    """
    Merge FOOOF features into spike DataFrame with paired current/previous window lists.
    
    Args:
        spike_data: DataFrame with spike parameters
        foof_results: List of FOOOF objects from compute_lfp_windows
        spike_to_window_map: Mapping from map_spikes_to_windows
        lfp_prefix: Prefix for LFP feature columns
        
    Returns:
        DataFrame with list-based features where:
        - current_* lists contain features from all windows containing the spike
        - previous_* lists contain features from preceding windows
        - List indices correspond to window pairs
    """
    # Input validation
    if 'spk_id' not in spike_data.columns:
        raise MissingColumnError("DataFrame must contain 'spk_id' column")

    df = spike_data.copy()
    
    # Initialize list-based columns
    feature_pairs = {
        'current': ['offset', 'exponent', 'r_squared', 'error', 'n_peaks'],
        'previous': ['offset', 'exponent', 'r_squared', 'error', 'n_peaks']
    }
    
    for timing in feature_pairs:
        for feat in feature_pairs[timing]:
            col_name = f"{lfp_prefix}{timing}_{feat}"
            df[col_name] = [[] for _ in range(len(df))]

    # Iterate through spikes and their associated windows
    for spk_id, window_indices in spike_to_window_map.items():
        row_idx = df.index[df['spk_id'] == spk_id]
        if row_idx.empty:
            warnings.warn(f"Spike ID {spk_id} not found in DataFrame", UserWarning)
            continue
            
        row_idx = row_idx[0]
        sorted_windows = sorted(window_indices)

        # Process each window and its temporal predecessor
        for win_idx in sorted_windows:
            # Add current window features
            if 0 <= win_idx < len(foof_results):
                current_fm = foof_results[win_idx]
                _append_features(df, row_idx, current_fm, f"{lfp_prefix}current")
                
                # Add previous window features if available
                if win_idx > 0 and (win_idx - 1) < len(foof_results):
                    prev_fm = foof_results[win_idx - 1]
                    _append_features(df, row_idx, prev_fm, f"{lfp_prefix}previous")
                else:
                    # Handle edge cases (first window)
                    _append_null_features(df, row_idx, f"{lfp_prefix}previous")
            else:
                warnings.warn(f"Invalid window index {win_idx} for spike {spk_id}", UserWarning)

    return df

def _append_features(
    df: pd.DataFrame,
    row_idx: int,
    foof_obj: FOOOF,
    prefix: str
) -> None:
    """Append FOOOF features to list columns"""
    try:
        df.at[row_idx, f"{prefix}_offset"].append(foof_obj.aperiodic_params_[0])
        df.at[row_idx, f"{prefix}_exponent"].append(foof_obj.aperiodic_params_[-1])
        df.at[row_idx, f"{prefix}_r_squared"].append(foof_obj.r_squared_)
        df.at[row_idx, f"{prefix}_error"].append(foof_obj.error_)
        df.at[row_idx, f"{prefix}_n_peaks"].append(foof_obj.n_peaks_)
    except AttributeError as e:
        warnings.warn(f"Missing FOOOF feature: {str(e)}", UserWarning)
        _append_null_features(df, row_idx, prefix)

def _append_null_features(
    df: pd.DataFrame,
    row_idx: int,
    prefix: str
) -> None:
    """Append None values to maintain list alignment"""
    df.at[row_idx, f"{prefix}_offset"].append(None)
    df.at[row_idx, f"{prefix}_exponent"].append(None)
    df.at[row_idx, f"{prefix}_r_squared"].append(None)
    df.at[row_idx, f"{prefix}_error"].append(None)
    df.at[row_idx, f"{prefix}_n_peaks"].append(None)





def compute_lfp_feature_means(
    df: pd.DataFrame,
    lfp_type: Literal["current", "previous"],
    drop_irrelevant: bool = True,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Compute the mean of LFP features (current or previous) and optionally drop irrelevant/original columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing LFP features.
        lfp_type (Literal["current", "previous"]): Whether to process current or previous LFP features.
        drop_irrelevant (bool): Whether to drop the irrelevant LFP features (e.g., drop previous when processing current).
        drop_original (bool): Whether to drop the original LFP list columns after computing their means.

    Returns:
        pd.DataFrame: A processed DataFrame with mean LFP features and optionally dropped columns.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_processed = df.copy()

    # Define LFP features based on type
    if lfp_type == "current":
        lfp_features = [
            'lfp_current_offset',
            'lfp_current_exponent',
            'lfp_current_r_squared',
            'lfp_current_error',
            'lfp_current_n_peaks'
        ]
        irrelevant_features = [
            'lfp_previous_offset',
            'lfp_previous_exponent',
            'lfp_previous_r_squared',
            'lfp_previous_error',
            'lfp_previous_n_peaks'
        ]
    elif lfp_type == "previous":
        lfp_features = [
            'lfp_previous_offset',
            'lfp_previous_exponent',
            'lfp_previous_r_squared',
            'lfp_previous_error',
            'lfp_previous_n_peaks'
        ]
        irrelevant_features = [
            'lfp_current_offset',
            'lfp_current_exponent',
            'lfp_current_r_squared',
            'lfp_current_error',
            'lfp_current_n_peaks'
        ]
    else:
        raise ValueError("lfp_type must be 'current' or 'previous'.")

    # Compute the mean of the two windows for each LFP feature
    for feature in lfp_features:
        new_col_name = f"{feature}_mean"
        df_processed[new_col_name] = df_processed[feature].apply(
            lambda x: (
                sum([v for v in x if v is not None]) / len([v for v in x if v is not None]) 
                if isinstance(x, list) and len([v for v in x if v is not None]) > 0 
                else None
            )
        )

    # Drop rows where all LFP means are NaN
    df_processed.dropna(
        subset=[f"{feature}_mean" for feature in lfp_features],
        how='all',  # Drop rows only if ALL LFP means are NaN
        inplace=True
    )

    # Drop original LFP list columns (optional)
    if drop_original:
        df_processed.drop(columns=lfp_features, inplace=True, errors='ignore')

    # Drop irrelevant LFP features (optional)
    if drop_irrelevant:
        df_processed.drop(columns=irrelevant_features, inplace=True, errors='ignore')

    return df_processed