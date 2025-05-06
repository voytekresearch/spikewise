"""
lfp_spike_window_analysis.py - Comprehensive module for LFP-spike analysis integration
"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Tuple, Dict, Union, Literal
from neurodsp import spectral
from fooof import FOOOF
import matplotlib.pyplot as plt
import mne
from tqdm.notebook import tqdm
import hashlib
from itertools import product

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


# ------------------------------ Core LFP Function ------------------------------
def extract_peak_features(fm: FOOOF, max_peaks: int) -> Dict:
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
        feats[f"{band}_peak_bw"] = np.mean(bws) if cfs else np.nan

    return feats



def compute_lfp_windows(
    lfp_signal: np.ndarray,
    fs: float,
    method: str = "welch",
    window_length_sec: int = 25,
    step_size_sec: float = 12.5,
    freq_range: Tuple[float, float] = (1, 90),
    fooof_params: Dict = None,
    plot: bool = False,
    min_overlap_percent: float = 50.0,
    decim_factor: int = 1,
    n_freqs: int = 50,
    time_bandwidth: float = 4.0,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[FOOOF], List[Tuple[int, int]], pd.DataFrame]:

    if method not in ["welch", "multitaper"]:
        raise ValueError("Method must be 'welch' or 'multitaper'.")

    window_length = int(window_length_sec * fs)
    step_size = int(step_size_sec * fs)

    window_times = [
        (start, start + window_length)
        for start in range(0, len(lfp_signal) - window_length + 1, step_size)
    ]

    spectra = []
    foof_results = []
    summary_records = []
    max_peaks = 3



    # ------------------ Welch Method ------------------
    if method == "welch":
        for start, end in window_times:
            segment = lfp_signal[start:end]
            try:
                fxx, pxx = spectral.compute_spectrum(
                    segment, fs, method="welch", window="hann", nperseg=int(fs * 4)
                )
            except Exception as e:
                raise SpectralComputationError(f"Welch error: {str(e)}")

            fm = FOOOF(**(fooof_params or {"max_n_peaks": 4, "verbose": False}))
            try:
                fm.fit(fxx, pxx, freq_range=freq_range)
                if plot:
                    fm.plot(plot_peaks="shade")
            except Exception as e:
                raise FOOOFFitError(f"FOOOF fit failed: {e}") from e

            spectra.append((fxx, pxx))
            foof_results.append(fm)

            summary = {
                "window_start": start,
                "window_end": end,
                "aperiodic_offset": fm.aperiodic_params_[0] if fm.has_model else np.nan,
                "aperiodic_exponent": fm.aperiodic_params_[1] if fm.has_model else np.nan,
                "r_squared": fm.r_squared_ if fm.has_model else np.nan,
                "n_peaks": len(fm.peak_params_) if fm.has_model else 0,
            }
            summary.update(extract_peak_features(fm, max_peaks))
            summary_records.append(summary)

    # ------------------ Multitaper Method ------------------
    elif method == "multitaper":
        epochs_array = np.array([lfp_signal[start:end] for start, end in window_times])
        epochs_array = np.expand_dims(epochs_array, axis=1)

        freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
        n_cycles = freqs * 1

        try:
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
            fxx = freqs
            tfr_arr = np.squeeze(np.swapaxes(tfr, 2, 3))
            if tfr_arr.ndim != 3:
                raise ValueError(f"Unexpected TFR shape: {tfr_arr.shape}")
        except Exception as e:
            raise SpectralComputationError(f"Multitaper error: {str(e)}") from e

        for i, psd in enumerate(tfr_arr):
            fm = FOOOF(**(fooof_params or {"max_n_peaks": 3, "verbose": False}))
            try:
                mean_psd = psd.mean(axis=0)
                fm.fit(fxx, mean_psd, freq_range=freq_range)
                if plot:
                    fm.plot(plot_peaks="shade")
            except Exception as e:
                raise FOOOFFitError(f"FOOOF fit failed: {e}") from e

            spectra.append((fxx, mean_psd))
            foof_results.append(fm)

            summary = {
                "window_start": window_times[i][0],
                "window_end": window_times[i][1],
                "aperiodic_offset": fm.aperiodic_params_[0] if fm.has_model else np.nan,
                "aperiodic_exponent": fm.aperiodic_params_[1] if fm.has_model else np.nan,
                "r_squared": fm.r_squared_ if fm.has_model else np.nan,
                "n_peaks": len(fm.peak_params_) if fm.has_model else 0,
            }
            summary.update(extract_peak_features(fm, max_peaks))
            summary_records.append(summary)

    summary_df = pd.DataFrame(summary_records)
    return spectra, foof_results, window_times, summary_df





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



def _append_features(df: pd.DataFrame, row_idx: int, fm: FOOOF, prefix: str) -> None:
    try:
        df.at[row_idx, f"{prefix}_offset"].append(fm.aperiodic_params_[0])
        df.at[row_idx, f"{prefix}_exponent"].append(fm.aperiodic_params_[-1])
        df.at[row_idx, f"{prefix}_r_squared"].append(fm.r_squared_)
        df.at[row_idx, f"{prefix}_error"].append(fm.error_)
        df.at[row_idx, f"{prefix}_n_peaks"].append(fm.n_peaks_)

        from fooof.analysis import get_band_peak_fg
        for band, frange in FREQ_BANDS.items():
            peak = get_band_peak_fg(fm, frange, select_highest=True)
            if peak is not None:
                cf, pw, bw = peak
            else:
                cf, pw, bw = None, None, None
            df.at[row_idx, f"{prefix}_{band}_peak_power"].append(pw)
            df.at[row_idx, f"{prefix}_{band}_peak_cf"].append(cf)
            df.at[row_idx, f"{prefix}_{band}_peak_bw"].append(bw)

    except Exception:
        _append_null_features(df, row_idx, prefix)

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
            "aperiodic_mode": mode,
            "peak_width_limits": (2.0, 8.0)  # 🔒 fixed range
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
) -> Tuple[pd.DataFrame, Dict[str, List[FOOOF]]]:
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
            _, foof_results, _, summary_df = compute_lfp_windows(
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

            summary_df["config_id"] = config["config_id"]
            for key, val in config.items():
                if key != "peak_width_limits":
                    summary_df[key] = val


            # Filtering
            valid_idx = summary_df["r_squared"] >= 0.8
            if config["aperiodic_mode"] == "knee":
                valid_idx &= (summary_df["aperiodic_offset"] >= freq_range[0]) & \
                             (summary_df["aperiodic_offset"] <= freq_range[1])

            summary_df = summary_df[valid_idx]
            valid_foofs = [f for f, keep in zip(foof_results, valid_idx) if keep]

            all_results.append(summary_df)
            foof_by_config[config["config_id"]] = valid_foofs

        except Exception as e:
            if verbose:
                print(f"[SKIPPED] {config['config_id']} due to: {e}")

    combined_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    return combined_df, foof_by_config
