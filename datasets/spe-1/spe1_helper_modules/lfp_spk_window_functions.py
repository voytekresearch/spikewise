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
    # [Previous implementation unchanged except error types]
    # ... (rest of create_lfp_windows implementation remains the same)

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
    # [Previous implementation unchanged]
    # ... (rest of map_spikes_to_windows implementation remains the same)

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
    # [Previous implementation unchanged]
    # ... (rest of combine_spike_lfp_features implementation remains the same)

# Helper Functions -------------------------------------------------------------
def _compute_spectrum(signal: np.ndarray, fs: float, **kwargs):
    """Internal spectrum computation (implementation unchanged)"""
    
def _add_fooof_features(df: pd.DataFrame, row_idx: int, foof_obj: FOOOF, prefix: str):
    """Internal feature adder (implementation unchanged)"""