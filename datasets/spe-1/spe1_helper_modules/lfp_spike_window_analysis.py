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
from fooof import FOOOFGroup, Bands


import mne

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
    freq_range: Tuple[float, float] = (1, 90),
    n_freqs: int = 100,
    time_window_len: float = 1.0,
    time_bandwidth: float = 4.0,
    n_peaks: int = 4,
    peak_width_lims: Tuple[float, float] = (1.0, 6.0),
) -> pd.DataFrame:
    """
    Perform sliding window LFP analysis with FOOOF spectral parameterization
    Includes comprehensive error handling and debug prints
    """
    
    # Initial parameters debug print
    print("\n=== Initial Parameters ===")
    print(f"Signal length: {len(lfp_signal)/fs:.1f}s ({len(lfp_signal)} samples)")
    print(f"Window: {window_length_sec}s ({int(window_length_sec*fs)} samples)")
    print(f"Step: {step_size_sec}s ({int(step_size_sec*fs)} samples)")
    print(f"Frequency range: {freq_range[0]}-{freq_range[1]}Hz")
    print(f"FOOOF peaks: {n_peaks}, Width limits: {peak_width_lims}")

    # Convert time parameters to samples
    window_length = int(window_length_sec * fs)
    step_size = int(step_size_sec * fs)
    n_samples = len(lfp_signal)

    # Create window indices
    try:
        starts = np.arange(0, n_samples - window_length + 1, step_size)
        ends = starts + window_length
        window_times = list(zip(starts, ends))
        n_windows = len(window_times)
        print(f"\nCreated {n_windows} windows")
    except Exception as e:
        print(f"\nError creating windows: {str(e)}")
        raise

    # Create epochs array
    try:
        epochs = np.stack([lfp_signal[start:end] for start, end in window_times])
        epochs = epochs[:, np.newaxis, :]  # Add channel dimension
        print(f"Epochs array shape: {epochs.shape}")
    except Exception as e:
        print(f"\nError creating epochs array: {str(e)}")
        raise

    # Compute multitaper TFR
    try:
        print("\n=== Computing Power Spectra ===")
        freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
        n_cycles = freqs * time_window_len
        print(f"Frequency bins: {n_freqs}")
        print(f"Cycle range: {n_cycles[0]:.1f}-{n_cycles[-1]:.1f} cycles")

        tfr = mne.time_frequency.tfr_array_multitaper(
            epochs,
            sfreq=fs,
            freqs=freqs,
            n_cycles=n_cycles,
            time_bandwidth=time_bandwidth,
            output='power',
            verbose=False
        )
        power_spectra = np.squeeze(tfr.mean(axis=-1))
        print(f"Power spectra shape: {power_spectra.shape}")
    except Exception as e:
        print(f"\nError in spectral computation: {str(e)}")
        raise

    # Configure FOOOF
    aperiodic_mode = 'knee' if time_window_len >= 0.5 else 'fixed'
    print("\n=== FOOOF Configuration ===")
    print(f"Aperiodic mode: {aperiodic_mode}")
    print(f"Max peaks: {n_peaks}")
    print(f"Peak width limits: {peak_width_lims}")

    # Initialize and fit FOOOFGroup
    fooof_grp = FOOOFGroup(
        peak_width_limits=peak_width_lims,
        max_n_peaks=n_peaks,
        aperiodic_mode=aperiodic_mode,
        verbose=False
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fooof_grp.fit(freqs, power_spectra, freq_range)
        print("\nFOOOF fitting completed successfully")
    except Exception as e:
        print(f"\nFOOOF fitting failed: {str(e)}")
        raise

    # Analyze fit results
    print("\n=== Fit Results ===")
    print(f"Total windows processed: {n_windows}")
    print(f"FOOOF models created: {len(fooof_grp)}")
    
    # Convert to numpy arrays for safe handling
    r_squared = np.array(fooof_grp.r_squared_)
    error = np.array(fooof_grp.error_)
    
    if r_squared.size > 0:
        success_mask = ~np.isnan(r_squared)
        print(f"Successful fits: {np.sum(success_mask)}")
        print(f"Failed fits: {np.sum(np.isnan(r_squared))}")
        print(f"Mean R²: {np.nanmean(r_squared):.2f}")
        print(f"Median error: {np.nanmedian(error):.2f}")
    else:
        print("No valid fit metrics available")

    # Initialize results dataframe
    results_df = pd.DataFrame({
        'window_start': starts,
        'window_end': ends,
        'window_duration': window_length_sec,
        'sample_rate': fs
    })

    # Safe parameter extraction functions
    def safe_aperiodic(params, idx: int, param_idx: int) -> float:
        """Safely extract aperiodic parameters with validation"""
        try:
            if idx < len(params):
                param_set = params[idx]
                if isinstance(param_set, np.ndarray) and len(param_set) > param_idx:
                    return param_set[param_idx]
        except (IndexError, TypeError, KeyError):
            pass
        return np.nan

    # Extract parameters
    print("\n=== Extracting Parameters ===")
    results_df['offset'] = [safe_aperiodic(fooof_grp.aperiodic_params_, i, 0) 
                          for i in range(n_windows)]
    results_df['exponent'] = [safe_aperiodic(fooof_grp.aperiodic_params_, i, -1) 
                            for i in range(n_windows)]
    
    if aperiodic_mode == 'knee':
        results_df['knee'] = [safe_aperiodic(fooof_grp.aperiodic_params_, i, 1) 
                            for i in range(n_windows)]

    # Add model metrics
    results_df['r_squared'] = [r_squared[i] if i < len(r_squared) else np.nan 
                              for i in range(n_windows)]
    results_df['error'] = [error[i] if i < len(error) else np.nan 
                          for i in range(n_windows)]

    # Add peak parameters
    try:
        print("Organizing peak parameters...")
        freq_bands = Bands({
            'delta': [1, 4],
            'theta': [4, 8],
            'alpha': [8, 12],
            'beta': [12, 30],
            'gamma': [30, 90]
        })
        peak_df = fooof_grp.to_df(freq_bands)
        results_df = pd.concat([results_df, peak_df], axis=1)
        print("Peak parameters added successfully")
    except Exception as e:
        print(f"Error organizing peak parameters: {str(e)}")
        # Create empty peak columns
        peak_cols = [f"{pre}_{band}_{n}" 
                    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']
                    for pre in ['CF', 'PW', 'BW'] 
                    for n in range(n_peaks)]
        for col in peak_cols:
            results_df[col] = np.nan

    # Final debug output
    print("\n=== Final Output ===")
    print(f"Result columns: {results_df.columns.tolist()}")
    print("First row sample:")
    print(results_df.iloc[0].to_dict())

    return results_df

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
            'lfp_current_n_peaks'
        ]
        irrelevant_features = [
            'lfp_previous_offset',
            'lfp_previous_exponent',
            'lfp_previous_r_squared',
            'lfp_previous_n_peaks',
            'lfp_current_error',
            'lfp_previous_error'
        ]
    elif lfp_type == "previous":
        lfp_features = [
            'lfp_previous_offset',
            'lfp_previous_exponent',
            'lfp_previous_r_squared',
            'lfp_previous_n_peaks'
        ]
        irrelevant_features = [
            'lfp_current_offset',
            'lfp_current_exponent',
            'lfp_current_r_squared',
            'lfp_current_n_peaks'
            'lfp_current_error',
            'lfp_previous_error'
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