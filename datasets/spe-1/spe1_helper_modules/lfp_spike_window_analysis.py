"""
lfp_spike_window_analysis.py - Comprehensive module for LFP-spike analysis integration
"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Tuple, Dict, Union, Literal
from neurodsp import spectral
from specparam import SpectralTimeModel
from specparam import SpectralModel
import matplotlib.pyplot as plt
import mne
import os 
from tqdm.notebook import tqdm
import hashlib
from itertools import product
from spe1_plotting import *

# ------------------------------ Custom Exceptions ------------------------------

class LFPSpikeWindowAnalysisError(Exception):
    pass

class SpectralComputationError(LFPSpikeWindowAnalysisError):
    pass

class FOOOFFitError(LFPSpikeWindowAnalysisError):
    pass

class MissingColumnError(LFPSpikeWindowAnalysisError):
    pass


# ------------------------------ Band Settings ------------------------------

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (15, 30),
    "gamma": (30, 90),
}



def assign_peak_to_band(cf: float) -> str:
    for band, (f_low, f_high) in BANDS.items():
        if f_low <= cf <= f_high:
            return band
    return "other"




def extract_peak_features(fm: SpectralModel, max_peaks: int) -> Dict:
    feats = {}
    band_features = {band: {"pw": [], "cf": [], "bw": []} for band in BANDS.keys()}

    if fm.has_model and fm.peak_params_ is not None:
        for i, (cf, pw, bw) in enumerate(fm.peak_params_[:max_peaks]):
            feats[f"peak_cf_{i}"] = cf
            feats[f"peak_pw_{i}"] = pw
            feats[f"peak_bw_{i}"] = bw

            band = assign_peak_to_band(cf)
            if band in band_features:
                band_features[band]["pw"].append(pw)
                band_features[band]["cf"].append(cf)
                band_features[band]["bw"].append(bw)

        for i in range(len(fm.peak_params_), max_peaks):
            feats[f"peak_cf_{i}"] = np.nan
            feats[f"peak_pw_{i}"] = np.nan
            feats[f"peak_bw_{i}"] = np.nan

    else:
        for i in range(max_peaks):
            feats[f"peak_cf_{i}"] = np.nan
            feats[f"peak_pw_{i}"] = np.nan
            feats[f"peak_bw_{i}"] = np.nan

    for band in BANDS.keys():
        pws = band_features[band]["pw"]
        cfs = band_features[band]["cf"]
        bws = band_features[band]["bw"]
        feats[f"{band}_peak_power"] = np.mean(pws) if pws else np.nan
        feats[f"{band}_peak_cf"] = np.mean(cfs) if cfs else np.nan
        feats[f"{band}_peak_bw"] = np.mean(bws) if bws else np.nan

    return feats



def compute_lfp_windows(
    lfp_signal: np.ndarray,
    fs: float,
    method: str = "welch",
    window_length_sec: int = 2,
    step_size_sec: float = 1,
    freq_range: Tuple[float, float] = (1, 90),
    fooof_params: Dict = None,
    decim_factor: int = 1,
    n_freqs: int = 50,
    time_bandwidth: float = 4.0,
    welch_params: Dict = None,
    multitaper_params: Dict = None,  
) -> Tuple[SpectralTimeModel, np.ndarray, List[Tuple[int, int]]]:

    if method not in ["welch", "multitaper"]:
        raise ValueError("Method must be 'welch' or 'multitaper'.")

    window_length = int(window_length_sec * fs)
    step_size = int(step_size_sec * fs)

    window_times = [
        (start, start + window_length)
        for start in range(0, len(lfp_signal) - window_length + 1, step_size)
    ]

    psd_list = []

    # ------------------ Welch Method ------------------
    if method == "welch":
        for start, end in window_times:
            segment = lfp_signal[start:end]
            fxx, pxx = spectral.compute_spectrum(
                segment,
                fs,
                method="welch",
                **(welch_params or {"window": "hann", "nperseg": int(fs)})
            )
            psd_list.append(pxx)
        freqs = fxx

    # ------------------ Multitaper Method ------------------
    elif method == "multitaper":
        # Override defaults with multitaper_params if provided
        if multitaper_params:
            n_freqs = multitaper_params.get("n_freqs", n_freqs)
            time_bandwidth = multitaper_params.get("time_bandwidth", time_bandwidth)
            decim_factor = multitaper_params.get("decim_factor", decim_factor)

        epochs_array = np.array([lfp_signal[start:end] for start, end in window_times])
        epochs_array = np.expand_dims(epochs_array, axis=1)

        freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
        n_cycles = freqs * 1

        tfr = mne.time_frequency.tfr_array_multitaper(
            epochs_array,
            sfreq=fs,
            freqs=freqs,
            n_cycles=n_cycles,
            time_bandwidth=time_bandwidth,
            output="power",
            decim=decim_factor,
            verbose=False,
        )
        tfr_arr = np.squeeze(np.swapaxes(tfr, 2, 3))  # shape (n_windows, n_times, n_freqs)
        psd_list = [psd.mean(axis=0) for psd in tfr_arr]  # average across time

    # Convert PSDs to 2D array
    powers = np.array(psd_list)  # shape (n_windows, n_freqs)
    powers_T = powers.T          # shape (n_freqs, n_windows) — required by SpectralTimeModel

    # --- Fit Specparam Time Model ---
    model = SpectralTimeModel(**(fooof_params or {"peak_width_limits": (4, 8), "max_n_peaks": 4}))
    model.fit(freqs, powers_T, freq_range=freq_range)

    return model, freqs, window_times


# ========================================
#Classify windows by gamma pw
# ========================================

def classify_gamma_windows(
    summary_df: pd.DataFrame,
    window_times: List[Tuple[int, int]],
    gamma_band_name: str = "gamma",
    high_percentile: float = 75,
    low_percentile: float = 25,
) -> Tuple[List[int], List[int]]:
    """
    Classify LFP windows as high or low gamma based on percentile thresholds.

    Args:
        summary_df : DataFrame from SpectralTimeModel.to_df()
        window_times : List of (start, end) sample indices per window
        gamma_band_name : The base name used for gamma power column (e.g., "gamma" → "gamma_pw")
        high_percentile : Percentile cutoff for high gamma windows
        low_percentile : Percentile cutoff for low gamma windows

    Returns:
        high_windows : List of window start times (in samples) classified as high gamma
        low_windows : List of window start times (in samples) classified as low gamma
    """
    gamma_powers = summary_df[f"{gamma_band_name}_pw"].values

    high_thresh = np.nanpercentile(gamma_powers, high_percentile)
    low_thresh = np.nanpercentile(gamma_powers, low_percentile)

    high_windows = [
        start for (start, _), pw in zip(window_times, gamma_powers) if pw >= high_thresh
    ]
    low_windows = [
        start for (start, _), pw in zip(window_times, gamma_powers) if pw <= low_thresh
    ]

    return high_windows, low_windows


# ============================================
#  Convert window inds into blocks
# ============================================
def merge_windows_into_blocks_flexible(
    window_starts: List[int],
    window_len_sec: float,
    fs: float,
    max_gap_sec: float = 1.0,
    min_block_len_sec: float = 4.0,
) -> List[Tuple[float, float]]:
    """
    Merge nearby windows into longer blocks, allowing for small gaps between them.

    Args:
        window_starts: List of window start times (in samples).
        window_len_sec: Length of each window in seconds.
        fs: Sampling frequency.
        max_gap_sec: Maximum gap (in seconds) allowed between adjacent windows to still be merged.
        min_block_len_sec: Minimum block duration to keep (in seconds).

    Returns:
        blocks: List of (start_time_sec, end_time_sec) tuples.
    """
    if not window_starts:
        return []

    window_starts = sorted(window_starts)
    max_gap_samples = int(max_gap_sec * fs)
    window_len_samples = int(window_len_sec * fs)

    blocks = []
    block_start = window_starts[0]
    block_end = block_start + window_len_samples

    for start in window_starts[1:]:
        if start - block_end <= max_gap_samples:
            # Extend the current block
            block_end = start + window_len_samples
        else:
            # Finalize current block
            duration = (block_end - block_start) / fs
            if duration >= min_block_len_sec:
                blocks.append((block_start / fs, block_end / fs))
            # Start new block
            block_start = start
            block_end = start + window_len_samples

    # Final block
    duration = (block_end - block_start) / fs
    if duration >= min_block_len_sec:
        blocks.append((block_start / fs, block_end / fs))

    return blocks

# ============================================
## Wrapper function for blocking analysis 
# ============================================

def segment_gamma_epochs(
    summary_df: pd.DataFrame,
    window_times: List[Tuple[int, int]],
    lfp_signal: np.ndarray,
    fs: float,
    window_len_sec: float = 2,
    gamma_band_name: str = "gamma",
    high_percentile: float = 77,
    low_percentile: float = 20,
    max_gap_sec: float = 1.0,
    min_block_len_sec: float = 5.0,
    visualize: bool = True,
) -> Tuple[List[int], List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Classify gamma windows and merge them into contiguous high/low gamma blocks.

    Returns:
        high_windows : List of window start sample indices classified as high gamma
        low_windows : List of window start sample indices classified as low gamma
        high_blocks : List of (start_sample, end_sample) tuples for high gamma blocks
        low_blocks : List of (start_sample, end_sample) tuples for low gamma blocks
    """
    # 1. Classify windows
    high_windows, low_windows = classify_gamma_windows(
        summary_df,
        window_times,
        gamma_band_name=gamma_band_name,
        high_percentile=high_percentile,
        low_percentile=low_percentile,
    )

    # 2. Merge windows into blocks
    high_blocks = merge_windows_into_blocks_flexible(
        high_windows, window_len_sec, fs,
        max_gap_sec=max_gap_sec,
        min_block_len_sec=min_block_len_sec
    )
    low_blocks = merge_windows_into_blocks_flexible(
        low_windows, window_len_sec, fs,
        max_gap_sec=max_gap_sec,
        min_block_len_sec=min_block_len_sec
    )

    # 3. Optional visualization
    if visualize:
        plot_lfp_with_blocks(
            lfp_signal=lfp_signal,
            fs=fs,
            high_blocks=high_blocks,
            low_blocks=low_blocks,
            title="Gamma Block Segmentation"
        )

    return high_windows, low_windows, high_blocks, low_blocks


def compute_block_overlap(blocks_a, blocks_b):
    """
    Compute the total overlap duration (in seconds) and overlapping intervals
    between two sets of blocks.

    Args:
        blocks_a, blocks_b : List of (start_sec, end_sec) tuples

    Returns:
        total_overlap : float, total overlapping time in seconds
        overlap_intervals : list of (start, end) overlap segments
    """
    overlaps = []
    total_overlap = 0.0
    
    for a_start, a_end in blocks_a:
        for b_start, b_end in blocks_b:
            overlap_start = max(a_start, b_start)
            overlap_end = min(a_end, b_end)
            if overlap_start < overlap_end:
                overlaps.append((overlap_start, overlap_end))
                total_overlap += (overlap_end - overlap_start)
    
    return total_overlap, overlaps



def save_lfp_blocks_to_bin(lfp_signal, fs, overlap_high, overlap_low, output_dir):
    """
    Save LFP segments for overlapping high and low gamma blocks as binary (.bin) files.

    Args:
        lfp_signal : 1D numpy array of the LFP signal
        fs : sampling frequency (Hz)
        overlap_high : list of (start_sec, end_sec) tuples for high gamma overlap segments
        overlap_low : list of (start_sec, end_sec) tuples for low gamma overlap segments
        output_dir : full path to the directory where .bin files will be saved
    """
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- High Gamma Blocks ---
    for i, (start_sec, end_sec) in enumerate(overlap_high):
        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)
        segment = lfp_signal[start_idx:end_idx].astype('float32')
        file_path = os.path.join(output_dir, f"high_block_{i}.bin")
        segment.tofile(file_path)

    # --- Low Gamma Blocks ---
    for i, (start_sec, end_sec) in enumerate(overlap_low):
        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)
        segment = lfp_signal[start_idx:end_idx].astype('float32')
        file_path = os.path.join(output_dir, f"low_block_{i}.bin")
        segment.tofile(file_path)

    print(f"✅ Saved {len(overlap_high)} high gamma blocks and {len(overlap_low)} low gamma blocks in:\n{output_dir}")

# =====================================

# -------------------------------------------------------------------
# Spike-LFP Mapping
# -------------------------------------------------------------------
def map_spikes_to_windows(
    spk_times_ms: List[float],
    spk_ids: List[int],
    window_times: List[Tuple[int, int]],
    df_spike_ids: Union[pd.Series, List[int]],
    fs: float
) -> Dict[int, List[int]]:
    window_times_ms = [(start / fs * 1000, end / fs * 1000) for (start, end) in window_times]
    return _map_spikes_to_window_helper(spk_times_ms, spk_ids, window_times_ms, df_spike_ids)

def _map_spikes_to_window_helper(
    spk_times_ms: List[float],
    spk_ids: List[int],
    window_times_ms: List[Tuple[float, float]],
    df_spike_ids: Union[pd.Series, List[int]]
) -> Dict[int, List[int]]:
    if len(spk_times_ms) != len(spk_ids):
        raise ValueError("spk_times_ms and spk_ids must match in length")
    valid_spike_ids = set(df_spike_ids)
    spike_to_window_map = {}
    for spk_id, spike_ms in zip(spk_ids, spk_times_ms):
        if spk_id not in valid_spike_ids:
            continue
        spike_to_window_map[spk_id] = []
        for window_idx, (start, end) in enumerate(window_times_ms):
            if start <= spike_ms < end or (spike_ms == end and window_idx < len(window_times_ms) - 1):
                spike_to_window_map[spk_id].append(window_idx)
    return spike_to_window_map

# -------------------------------------------------------------------
# Combine spike features with LFP window features
# -------------------------------------------------------------------
def combine_spike_lfp_features(
    spike_data: pd.DataFrame,
    summary_df: pd.DataFrame,
    spike_to_window_map: Dict[int, List[int]],
    lfp_prefix: str = "lfp_"
) -> pd.DataFrame:
    df = spike_data.copy()
    if 'spk_id' not in df.columns:
        raise ValueError("spk_id column missing from spike_data")

    # List of all band-based features
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_feats = [f"{band}_{kind}" for band in bands for kind in ['peak_power', 'peak_cf', 'peak_bw']]
    base_feats = ['offset', 'exponent', 'r_squared', 'error', 'n_peaks']
    all_feats = base_feats + band_feats

    # Initialize list-style columns
    for t in ["current", "previous"]:
        for f in all_feats:
            df[f"{lfp_prefix}{t}_{f}"] = [[] for _ in range(len(df))]

    for spk_id, window_idxs in spike_to_window_map.items():
        row_idx = df.index[df["spk_id"] == spk_id]
        if row_idx.empty:
            continue
        row_idx = row_idx[0]

        for win_idx in sorted(window_idxs):
            if win_idx < len(summary_df):
                current_row = summary_df.iloc[win_idx]
                _append_summary_features(df, row_idx, current_row, f"{lfp_prefix}current")

                if win_idx > 0:
                    prev_row = summary_df.iloc[win_idx - 1]
                    _append_summary_features(df, row_idx, prev_row, f"{lfp_prefix}previous")
                else:
                    _append_null_features(df, row_idx, f"{lfp_prefix}previous")
            else:
                _append_null_features(df, row_idx, f"{lfp_prefix}current")

    #Remove original generic peak features (cf/pw/bw if still there)
    peak_cols = [col for col in df.columns if any(x in col for x in ['peak_cf_', 'peak_pw_', 'peak_bw_'])]
    df.drop(columns=peak_cols, inplace=True, errors="ignore")

    #Drop band features where all values are None
    for col in df.columns:
        if isinstance(df[col].iloc[0], list) and all(
            (v is None or (isinstance(v, list) and all(x is None for x in v)))
            for v in df[col]
        ):
            df.drop(columns=col, inplace=True)

    return df

def _append_summary_features(df: pd.DataFrame, row_idx: int, row: pd.Series, prefix: str) -> None:
    base_feat_map = {
        'offset': 'aperiodic_offset',
        'exponent': 'aperiodic_exponent',
        'r_squared': 'r_squared',
        'error': 'error',
        'n_peaks': 'n_peaks',
    }

    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_feats = [f"{band}_{kind}" for band in bands for kind in ['peak_power', 'peak_cf', 'peak_bw']]
    all_feats = list(base_feat_map.keys()) + band_feats

    for feat in all_feats:
        col = f"{prefix}_{feat}"
        if feat in base_feat_map:
            val = row.get(base_feat_map[feat], None)
        else:
            val = row.get(feat, None)
        df.at[row_idx, col].append(val)


def _append_features(df: pd.DataFrame, row_idx: int, fm: SpectralModel, prefix: str) -> None:
    try:
        df.at[row_idx, f"{prefix}_offset"].append(fm.aperiodic_params_[0])
        df.at[row_idx, f"{prefix}_exponent"].append(fm.aperiodic_params_[-1])
        df.at[row_idx, f"{prefix}_r_squared"].append(fm.r_squared_)
        df.at[row_idx, f"{prefix}_error"].append(fm.error_)
        df.at[row_idx, f"{prefix}_n_peaks"].append(len(fm.peak_params_))

        for band, frange in FREQ_BANDS.items():
            peak = get_band_peak_manual(fm, frange)
            if peak is not None:
                cf, pw, bw = peak
            else:
                cf, pw, bw = None, None, None
            df.at[row_idx, f"{prefix}_{band}_peak_power"].append(pw)
            df.at[row_idx, f"{prefix}_{band}_peak_cf"].append(cf)
            df.at[row_idx, f"{prefix}_{band}_peak_bw"].append(bw)

    except Exception:
        _append_null_features(df, row_idx, prefix)

def get_band_peak_manual(fm: SpectralModel, band: Tuple[float, float]):
    """Manually extract the peak with highest power within a given band."""
    peaks = fm.peak_params_
    band_peaks = [peak for peak in peaks if band[0] <= peak[0] <= band[1]]
    if band_peaks:
        return max(band_peaks, key=lambda x: x[1])
    return None



def _append_null_features(df: pd.DataFrame, row_idx: int, prefix: str) -> None:
    base_feats = ['offset', 'exponent', 'r_squared', 'error', 'n_peaks']
    band_feats = [f"{band}_{kind}" for band in ['delta', 'theta', 'alpha', 'beta', 'gamma'] for kind in ['peak_power', 'peak_cf', 'peak_bw']]
    for f in base_feats + band_feats:
        df.at[row_idx, f"{prefix}_{f}"].append(None)

# -------------------------------------------------------------------
# Average list-style LFP features into mean features
# -------------------------------------------------------------------
def compute_lfp_feature_means(
    df: pd.DataFrame,
    lfp_type: Literal["current", "previous"],
    drop_irrelevant: bool = True,
    drop_original: bool = True
) -> pd.DataFrame:
    df_out = df.copy()

    base_feats = ['offset', 'exponent', 'r_squared', 'error', 'n_peaks']
    band_feats = [f"{band}_{kind}" for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']
                  for kind in ['peak_power', 'peak_cf', 'peak_bw']]
    feats = base_feats + band_feats

    for feat in feats:
        col = f"lfp_{lfp_type}_{feat}"
        if col in df_out.columns:
            df_out[f"{col}_mean"] = df_out[col].apply(
                lambda x: np.nanmean([v for v in x if v is not None]) if isinstance(x, list) and x else None
            )

    # Only keep columns that actually exist
    mean_cols = [f"lfp_{lfp_type}_{feat}_mean" for feat in feats]
    existing_mean_cols = [col for col in mean_cols if col in df_out.columns]
    df_out.dropna(subset=existing_mean_cols, how="all", inplace=True)

    if drop_original:
        df_out.drop(columns=[f"lfp_{lfp_type}_{feat}" for feat in feats], inplace=True, errors="ignore")

    if drop_irrelevant:
        other = "previous" if lfp_type == "current" else "current"
        to_drop = [col for col in df_out.columns if f"lfp_{other}_" in col]
        df_out.drop(columns=to_drop, inplace=True, errors="ignore")

    return df_out



# ==========================
# Sensitivity Analysis Code 
# ==========================


def make_fooof_param_grid(
    max_n_peaks_list: List[int],
    peak_threshold_list: List[float],
    aperiodic_modes: List[str]
) -> List[Dict]:
    return [
        {
            "max_n_peaks": n,
            "peak_threshold": t,
            "aperiodic_mode": mode
        }
        for n, t, mode in product(max_n_peaks_list, peak_threshold_list, aperiodic_modes)
    ]



def make_config_id(config: dict) -> str:
    """Generate a short hash for a config dictionary."""
    config_str = str(sorted(config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def sensitivity_analysis(
    lfp_signal: np.ndarray,
    fs: float,
    freq_range: Tuple[float, float],
    window_lengths: List[int],
    methods: List[str],
    time_bandwidths: List[float],
    fooof_param_grid: List[Dict],
    step_ratio: float = 0.5,
    n_freqs: int = 50,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, List]]:
    """
    Perform sensitivity analysis over various parameter combinations.

    Returns:
    - A DataFrame containing the summary of FOOOF fits.
    - A dictionary mapping config_id to the list of corresponding FOOOF model objects.
    """
    all_results = []
    foof_by_config = {}

    # Build full parameter grid
    param_grid = []
    for wl, method, tb, fooof_params in product(window_lengths, methods, time_bandwidths, fooof_param_grid):
        config = {
            "window_length_sec": wl,
            "step_size_sec": wl * step_ratio,
            "method": method,
            "time_bandwidth": tb,
            **fooof_params
        }
        config["config_id"] = make_config_id(config)
        param_grid.append(config)

    for config in tqdm(param_grid, desc="Param combos"):
        try:
            # Compute LFP windows and obtain FOOOF results
            _, _, foof_results, summary_df = compute_lfp_windows(
                lfp_signal=lfp_signal,
                fs=fs,
                method=config["method"],
                window_length_sec=config["window_length_sec"],
                step_size_sec=config["step_size_sec"],
                freq_range=freq_range,
                fooof_params={k: config[k] for k in fooof_param_grid[0].keys()},
                plot=False,
                n_freqs=n_freqs,
                time_bandwidth=config["time_bandwidth"]
            )

            # Apply filtering
            mask = summary_df["r_squared"] >= 0.8
            if config["aperiodic_mode"] == "knee":
                mask &= (
                    (summary_df["aperiodic_offset"] >= freq_range[0]) &
                    (summary_df["aperiodic_offset"] <= freq_range[1])
                )

            # Apply mask to summary_df and foof_results
            summary_df = summary_df[mask].reset_index(drop=True)
            foof_results = [f for f, keep in zip(foof_results, mask) if keep]

            # Add config info to summary_df
            summary_df["config_id"] = config["config_id"]
            for key, val in config.items():
                summary_df[key] = val

            all_results.append(summary_df)
            foof_by_config[config["config_id"]] = foof_results

        except Exception as e:
            if verbose:
                print(f"[SKIPPED] {config['config_id']} due to: {e}")

    if all_results:
        return pd.concat(all_results, ignore_index=True), foof_by_config
    else:
        return pd.DataFrame(), {}


