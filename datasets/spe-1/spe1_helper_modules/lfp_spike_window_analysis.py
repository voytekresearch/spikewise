"""
lfp_spike_window_analysis.py - Comprehensive module for LFP-spike analysis integration
"""

import numpy as np
import pandas as pd
import warnings
import mne
import os 
from typing import List, Tuple, Dict, Union, Literal
from neurodsp import spectral
from specparam import SpectralTimeModel
from specparam import SpectralModel
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import hashlib
from itertools import product
from scipy.stats import ttest_ind
from spe1_plotting import *

# ------------------------------------------------------------------------------------------- #
# ------------------------------ CODE FOR ALL LFP-PATCH METHODS ------------------------------
# ------------------------------------------------------------------------------------------- #

def windows_to_tuples(
    window_starts: list[int],
    window_len_sec: float,
    fs: float
) -> list[tuple[float, float]]:
    """
    Convert a list of window start sample indices to (start_sec, end_sec) tuples.

    Args:
        window_starts : list of window start times in samples
        window_len_sec : length of each window in seconds
        fs : sampling frequency

    Returns:
        List of (start_sec, end_sec) tuples
    """
    window_tuples = []
    for start in sorted(window_starts):
        start_sec = start / fs
        end_sec = start_sec + window_len_sec
        window_tuples.append((start_sec, end_sec))
    return window_tuples



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

def compare_spike_params_groups(spike_df, group_col, groups, params):
    """
    Compare spike parameters between two groups using t-tests and violin plots.

    Args:
        spike_df : pd.DataFrame
            DataFrame with spike parameters and a group label column.
        group_col : str
            Column name that contains the group labels.
        groups : tuple of str
            (group1, group2) where group2 can be 'rest' to include all spikes not in group1.
        params : list of str
            Parameter columns to compare.

    Returns:
        results : dict
            Dictionary with parameter -> (t_stat, p_value).
    """
    group1, group2 = groups

    #  Handle "rest" logic
    if group2 == "rest":
        df1 = spike_df[spike_df[group_col] == group1]
        df2 = spike_df[spike_df[group_col] != group1]
    else:
        df1 = spike_df[spike_df[group_col] == group1]
        df2 = spike_df[spike_df[group_col] == group2]

    results = {}

    for param in params:
        vals1 = df1[param].dropna()
        vals2 = df2[param].dropna()

        if len(vals1) == 0 or len(vals2) == 0:
            print(f"⚠️ Skipping {param} – one of the groups has no data.")
            continue

        # Welch's t-test
        t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False, nan_policy='omit')
        results[param] = (t_stat, p_val)

        # --- Violin Plot ---
        plt.figure(figsize=(5, 4))
        sns.violinplot(data=[vals1, vals2], cut=0)
        plt.xticks([0, 1], [group1, group2])
        plt.ylabel(param)
        plt.title(f"{param}\n t={t_stat:.2f}, p={p_val:.4f}")  # formatted p-value
        plt.tight_layout()
        plt.show()

    return results


def compute_lfp_feature_means(
    df: pd.DataFrame,
    lfp_type: Literal["current", "previous"],
    drop_irrelevant: bool = True,
    drop_original: bool = True
) -> pd.DataFrame:
    df_out = df.copy()

    """ # Average list-style LFP features into mean features"""

    base_feats = ['offset', 'exponent', 'r_squared', 'error', 'n_peaks']
    band_feats = [f"{band}_{kind}" for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']
                  for kind in ['peak_power', 'peak_cf', 'peak_bw']]
    feats = base_feats + band_feats

    for feat in feats:
        col = f"lfp_{lfp_type}_{feat}"
        if col in df_out.columns:
            df_out[f"{col}_mean"] = df_out[col].apply(
                # minimal change: guard nanmean so we don't call it on an effectively empty list
                lambda x: (
                    np.nanmean([v for v in x if v is not None])
                    if isinstance(x, list) and x and any(v is not None for v in x)
                    else (np.nan if isinstance(x, list) else None)
                )
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




# ------------------------------------------------------------------------------------------- #
# ------------------CODE CLASSIFICIATION BLOCKING METHODS 1) AND 2)   ----------------------
# ------------------------------------------------------------------------------------------- #




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

    print(f"Saved {len(overlap_high)} high gamma blocks and {len(overlap_low)} low gamma blocks in:\n{output_dir}")




def label_spikes(
    spike_df: pd.DataFrame,
    blocks_a: List[Tuple[float, float]],
    label_a: str,
    blocks_b: Optional[List[Tuple[float, float]]] = None,
    label_b: Optional[str] = None,
    default_label: str = "none",
    time_col: str = "spk_times_ms",
    priority: str = "a"  # {"a","b","first"} — which label wins if a spike is in both
) -> pd.DataFrame:
    """
    Label spikes based on time blocks.

    Modes
    -----
    1) Two-label mode:
       - Provide blocks_a + label_a AND blocks_b + label_b.
       - Spikes in A -> label_a; in B -> label_b; in neither -> default_label.
       - If a spike falls in BOTH, use `priority`.

    2) Label-vs-rest/none mode:
       - Provide only blocks_a + label_a (leave blocks_b=None / label_b=None).
       - Spikes in A -> label_a; everything else -> default_label (e.g., "rest").

    Parameters
    ----------
    spike_df : DataFrame with spike times column in ms.
    blocks_a, blocks_b : lists of (start_sec, end_sec).
    label_a, label_b : labels to assign.
    default_label : label for spikes in neither (or the 'rest' label for mode 2).
    time_col : name of spike time column in ms.
    priority : "a", "b", or "first":
        - "a": A wins if spike in both A and B
        - "b": B wins if spike in both
        - "first": whichever block list hits first in the check order (A then B)

    Returns
    -------
    DataFrame with new column 'method_block_label'.
    """

    def in_block(t: float, blocks: List[Tuple[float, float]]) -> bool:
        # blocks are in SECONDS; t is in SECONDS
        return any(s <= t < e for s, e in blocks)

    spikes = spike_df.copy()
    spike_times_sec = spikes[time_col].to_numpy(dtype=float) / 1000.0
    labels = np.full(len(spikes), default_label, dtype=object)

    two_label_mode = blocks_b is not None and label_b is not None

    if not two_label_mode:
        # ----- Label-vs-rest -----
        for i, t in enumerate(spike_times_sec):
            if in_block(t, blocks_a):
                labels[i] = label_a
    else:
        # ----- Two-label mode -----
        for i, t in enumerate(spike_times_sec):
            in_a = in_block(t, blocks_a)
            in_b = in_block(t, blocks_b)

            if in_a and in_b:
                if priority == "a":
                    labels[i] = label_a
                elif priority == "b":
                    labels[i] = label_b
                else:  # "first" => A checked first, so label_a if in_a else label_b
                    labels[i] = label_a
            elif in_a:
                labels[i] = label_a
            elif in_b:
                labels[i] = label_b
            # else: keep default_label

    spikes["method_block_label"] = labels
    return spikes



# ------------------------------------------------------------------------------------------- #
# ------------------1) CODE FOR LARGE LFP WINDOW GAMMA CLASSIFICATION  ----------------------
# ------------------------------------------------------------------------------------------- #

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

#Functions to compare windowing specparam results

def compute_method_comparison_blocks(
    welch_high, welch_low, mt_high, mt_low,
    mode="overlap"
):
    """
     Compute blocks from method comparison (overlap / non-contradictory)

    Parameters
    ----------
    welch_high, welch_low : list of (start, end)
        Welch high/low gamma blocks (sec)
    mt_high, mt_low : list of (start, end)
        Multitaper high/low gamma blocks (sec)
    mode : str
        "overlap" → AND logic (only blocks present in both methods)
        "noncontradictory" → OR logic minus conflicts

    Returns
    -------
    comp_high_blocks, comp_low_blocks : list of (start, end)
    info_label : str
        Description of the comparison logic
    """

    def intersect_blocks(blocks_a, blocks_b):
        """Return intersections (AND) between two block lists."""
        overlaps = []
        for a_start, a_end in blocks_a:
            for b_start, b_end in blocks_b:
                s, e = max(a_start, b_start), min(a_end, b_end)
                if s < e:
                    overlaps.append((s, e))
        return overlaps

    def union_noncontradictory(high_a, low_a, high_b, low_b):
        """Return high blocks from either method while cutting out overlaps with any low blocks."""
        combined_highs = sorted(high_a + high_b)
        combined_lows = sorted(low_a + low_b)
        cleaned = []
        for hs, he in combined_highs:
            segments = [(hs, he)]
            for ls, le in combined_lows:
                new_segments = []
                for seg_s, seg_e in segments:
                    if le <= seg_s or ls >= seg_e:
                        new_segments.append((seg_s, seg_e))
                    else:
                        if seg_s < ls:
                            new_segments.append((seg_s, ls))
                        if le < seg_e:
                            new_segments.append((le, seg_e))
                segments = new_segments
            cleaned.extend(segments)
        return cleaned

    # --- Choose comparison logic ---
    if mode == "overlap":
        comp_high_blocks = intersect_blocks(welch_high, mt_high)
        comp_low_blocks  = intersect_blocks(welch_low, mt_low)
        info_label = "Overlap (AND logic)"
    elif mode == "noncontradictory":
        comp_high_blocks = union_noncontradictory(welch_high, welch_low, mt_high, mt_low)
        comp_low_blocks  = union_noncontradictory(welch_low, welch_high, mt_low, mt_high)
        info_label = "Non-Contradictory (OR minus conflicts)"
    else:
        raise ValueError("mode must be 'overlap' or 'noncontradictory'")

    return comp_high_blocks, comp_low_blocks, info_label



def get_and_plot_method_comparison_blocks(
    lfp_signal, fs,
    welch_high, welch_low, mt_high, mt_low,
    mode="overlap",
    plot=True
):
    """
    Wrapper to compute and optionally plot gamma blocks from Welch vs Multitaper comparison.

    Returns
    -------
    comp_high_blocks, comp_low_blocks
    """
    comp_high, comp_low, info = compute_method_comparison_blocks(
        welch_high, welch_low, mt_high, mt_low, mode=mode
    )

    if plot:
        plot_gamma_blocks_generic(lfp_signal, fs, comp_high, comp_low, info)

    return comp_high, comp_low


def segment_gamma_epochs(
    summary_df: pd.DataFrame,
    window_times: List[Tuple[int, int]],
    lfp_signal: np.ndarray,
    fs: float,
    window_len_sec: float = 2,
    gamma_band_name: str = "gamma",
    high_percentile: float = 80,
    low_percentile: float = 10,
    max_gap_sec: float = 1.0,
    min_block_len_sec: float = 5.0,
    visualize: bool = True,
) -> Tuple[List[int], List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Wrapper function for blocking analysis 

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
        plot_gamma_blocks_generic(
            lfp_signal=lfp_signal,
            fs=fs,
            high_blocks=high_blocks,
            low_blocks=low_blocks,
            title="Gamma Block Segmentation"
        )

    return high_windows, low_windows, high_blocks, low_blocks




# ------------------------------------------------------------------------------------------- #
# -------------------------2) CODE GAMMA BURST CLASSIFICATION  ---------------------------------
# ------------------------------------------------------------------------------------------- #

def detect_gamma_bursts(
    gamma_amp: np.ndarray,
    lfp_times: np.ndarray,
    high_thresh_percentile: float = 80,
    low_thresh_percentile: float = 20,
    min_duration_ms: int = 50
) -> dict:
    """
    Detect high and low gamma bursts based on analytic amplitude thresholds.

    Parameters
    ----------
    gamma_amp : np.ndarray
        Amplitude of gamma-filtered LFP (same length as lfp_times).
    lfp_times : np.ndarray
        Time vector for the LFP in milliseconds.
    high_thresh_percentile : float
        Percentile threshold for defining high gamma bursts.
    low_thresh_percentile : float
        Percentile threshold for defining low gamma bursts.
    min_duration_ms : int
        Minimum burst duration to include (milliseconds).

    Returns
    -------
    dict
        {
            'high_bursts': list of (start_ms, end_ms),
            'low_bursts': list of (start_ms, end_ms),
            'rest_bursts': list of (start_ms, end_ms),
            'high_mask': boolean array,
            'low_mask': boolean array
        }
    """
    amp = gamma_amp
    times = lfp_times

    # Thresholds
    high_thresh = np.nanpercentile(amp, high_thresh_percentile)
    low_thresh = np.nanpercentile(amp, low_thresh_percentile)

    def get_mask(thresh_type):
        if thresh_type == 'high':
            return amp >= high_thresh
        elif thresh_type == 'low':
            return amp <= low_thresh
        else:
            return np.logical_and(amp < high_thresh, amp > low_thresh)

    def extract_bursts(mask, min_len):
        bursts = []
        in_burst = False
        start = None
        for i, val in enumerate(mask):
            if val and not in_burst:
                start = times[i]
                in_burst = True
            elif not val and in_burst:
                end = times[i]
                if (end - start) >= min_len:
                    bursts.append((start, end))
                in_burst = False
        if in_burst and (times[-1] - start) >= min_len:
            bursts.append((start, times[-1]))
        return bursts

    high_mask = get_mask('high')
    low_mask = get_mask('low')
    rest_mask = get_mask('rest')

    return {
        'high_bursts': extract_bursts(high_mask, min_duration_ms),
        'low_bursts': extract_bursts(low_mask, min_duration_ms),
        'rest_bursts': extract_bursts(rest_mask, min_duration_ms),
        'high_mask': high_mask,
        'low_mask': low_mask
    }


def merge_burst_intervals_into_blocks(
    burst_intervals,
    fs: float,
    max_gap_sec: float = 1.0,
    min_block_len_sec: float = 4.0,
    units: str = "auto",  # "auto", "seconds", "samples"
):
    """
    Merge gamma burst intervals into longer blocks, allowing small gaps.

    Parameters
    ----------
    burst_intervals : list[tuple]
        List of (start, end) for bursts. Can be in seconds or samples.
    fs : float
        Sampling rate (Hz).
    max_gap_sec : float
        Max gap between adjacent bursts (in seconds) to still be merged.
    min_block_len_sec : float
        Minimum duration (in seconds) to keep a merged block.
    units : {"auto","seconds","samples"}
        - "seconds": treat burst_intervals as seconds.
        - "samples": treat burst_intervals as samples and convert to seconds.
        - "auto": detect based on magnitude/type.

    Returns
    -------
    list[tuple[float,float]]
        Merged (start_sec, end_sec) blocks in seconds.
    """
    if not burst_intervals:
        return []

    # --- Normalize to seconds ---
    norm = []
    for s, e in burst_intervals:
        if e < s:
            s, e = e, s

        if units == "seconds":
            s_sec, e_sec = float(s), float(e)
        elif units == "samples":
            s_sec, e_sec = float(s) / fs, float(e) / fs
        else:  # auto
            is_samples = (
                isinstance(s, (int, np.integer)) and isinstance(e, (int, np.integer))
            ) or (s > 1e5 or e > 1e5)
            if is_samples:
                s_sec, e_sec = float(s) / fs, float(e) / fs
            else:
                s_sec, e_sec = float(s), float(e)

        norm.append((s_sec, e_sec))

    # --- Sort & merge with gap rule ---
    norm.sort(key=lambda x: x[0])

    merged = []
    cur_s, cur_e = norm[0]
    for s, e in norm[1:]:
        if s <= cur_e + max_gap_sec:  # merge if within allowed gap
            cur_e = max(cur_e, e)
        else:
            if (cur_e - cur_s) >= min_block_len_sec:
                merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    if (cur_e - cur_s) >= min_block_len_sec:
        merged.append((cur_s, cur_e))

    return merged



# ------------------------------------------------------------------------------------------- #
# ------------------3) CODE FOR TIME RESOLVED SPECPARAM   ----------------------
# ------------------------------------------------------------------------------------------- #
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


def _append_null_features(df: pd.DataFrame, row_idx: int, prefix: str) -> None:
    base_feats = ['offset', 'exponent', 'r_squared', 'error', 'n_peaks']
    band_feats = [f"{band}_{kind}" for band in ['delta', 'theta', 'alpha', 'beta', 'gamma'] for kind in ['peak_power', 'peak_cf', 'peak_bw']]
    for f in base_feats + band_feats:
        df.at[row_idx, f"{prefix}_{f}"].append(None)




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



# ------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------- #












