import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from spikeparam_plotting import *
from specparam import SpectralTimeModel
import matplotlib.patches as mpatches



def plot_spike_time_histogram(
    spk_times_ms: np.ndarray,
    window_times: List[Tuple[int, int]],  # List of (start_sample, end_sample) tuples
    lfp_fs: float,  # LFP sampling rate (Hz)
    bins: int = 300,
    spike_color: str = "blue",
    window_color: str = "orange",
    alpha: float = 0.7,
    title: str = "Spike Time Distribution with LFP Windows",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a histogram of spike times with overlaid LFP windows.
    
    Parameters:
        spk_times_ms (np.ndarray): Array of spike times in milliseconds.
        window_times (List[Tuple[int, int]]): List of (start, end) tuples for LFP windows (in samples).
        lfp_fs (float): Sampling rate of the LFP signal (Hz).
        bins (int): Number of bins for the histogram.
        spike_color (str): Color for the spike time histogram.
        window_color (str): Color for the LFP window overlays.
        alpha (float): Transparency for the histogram and windows.
        title (str): Title of the plot.
        save_path (Optional[str]): Path to save the plot (e.g., "plot.png"). If None, plot is displayed.
    """
    plt.figure(figsize=(12, 6))
    plt.hist(spk_times_ms, bins=bins, color=spike_color, alpha=alpha, edgecolor='black', label="Spike Times")

    # Plot LFP windows
    for idx, (start, end) in enumerate(window_times):
        start_ms = start / lfp_fs * 1000  # Convert to milliseconds
        end_ms = end / lfp_fs * 1000      # Convert to milliseconds
        plt.axvspan(start_ms, end_ms, color=window_color, alpha=0.2, label='LFP Window' if idx == 0 else "")
        
        # Add vertical lines (optional)
        plt.axvline(start_ms, color=window_color, linestyle='--', alpha=0.5)
        plt.axvline(end_ms, color=window_color, linestyle='--', alpha=0.5)

    plt.xlabel("Time (ms)")
    plt.ylabel("Spike Count")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_lfp_spk_correlation_heatmap(
    df: pd.DataFrame,
    spike_features: List[str],
    lfp_features: List[str],
    title: str = "Correlation Heatmap",
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "coolwarm",
    annot: bool = True,
    fmt: str = ".2f",
    calculate_corr: bool =True,
    show_sig: bool=True
    ):
    """
    Plot a correlation heatmap between spike features and LFP features with significance stars.

    Parameters:
        df (pd.DataFrame): DataFrame containing spike and LFP features.
        spike_features (List[str]): List of spike feature column names.
        lfp_features (List[str]): List of LFP feature column names.
        title (str): Title of the plot.
        figsize (Tuple[int, int]): Size of the plot (width, height).
        cmap (str): Color map for the heatmap.
        annot (bool): Whether to annotate the heatmap with correlation values.
        fmt (str): Format for the annotations (e.g., ".2f" for 2 decimal places).
        save_path (Optional[str]): Path to save the plot (e.g., "heatmap.png"). If None, plot is displayed.
    """

    correlation_matrix = df[spike_features + lfp_features].corr()
    spike_vs_lfp_corr = correlation_matrix.loc[spike_features, lfp_features]


    # Plot with significance stars
    plot_corr_heatmap(
        df_features=df,
        spike_features=spike_features,
        lfp_features=lfp_features,
        calculate_corr=calculate_corr,
        show_sig=show_sig, title = title 
    )


def plot_avg_spectra(model: SpectralTimeModel, freqs: np.ndarray, high_inds: List[int], low_inds: List[int]):
    spectra = model.spectrogram.T  # shape: (n_windows, n_freqs)
    avg_high = np.nanmean(spectra[high_inds], axis=0)
    avg_low = np.nanmean(spectra[low_inds], axis=0)
    std_high = np.nanstd(spectra[high_inds], axis=0)
    std_low = np.nanstd(spectra[low_inds], axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, avg_high, label=f"High Gamma (n={len(high_inds)})", color='red')
    plt.fill_between(freqs, avg_high - std_high, avg_high + std_high, color='red', alpha=0.3)
    plt.plot(freqs, avg_low, label=f"Low Gamma (n={len(low_inds)})", color='blue')
    plt.fill_between(freqs, avg_low - std_low, avg_low + std_low, color='blue', alpha=0.3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (log10)')
    plt.title('Average Spectra: High vs Low Gamma Windows')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_lfp_with_blocks(
    lfp_signal: np.ndarray,
    fs: float,
    high_blocks: List[Tuple[float, float]],
    low_blocks: List[Tuple[float, float]],
    title: str = "LFP Signal with Gamma Blocks"
):
    """
    Plot the LFP signal and overlay high/low gamma blocks.

    Args:
        lfp_signal : 1D LFP time series
        fs : Sampling frequency in Hz
        high_blocks : List of (start, end) for high gamma blocks (samples or seconds)
        low_blocks : List of (start, end) for low gamma blocks (samples or seconds)
        title : Plot title
    """
    def convert_to_samples(blocks):
        return [
            (int(start * fs), int(end * fs)) if start < 100000 and end < 100000 else (start, end)
            for start, end in blocks
        ]

    # Convert to sample indices if needed
    high_blocks_s = convert_to_samples(high_blocks)
    low_blocks_s = convert_to_samples(low_blocks)

    # Plot
    time = np.arange(len(lfp_signal)) / fs
    plt.figure(figsize=(15, 5))
    plt.plot(time, lfp_signal, color='black', linewidth=0.8, label='LFP Signal')

    for start, end in high_blocks_s:
        plt.axvspan(start / fs, end / fs, color='red', alpha=0.3, label='High Gamma')
    for start, end in low_blocks_s:
        plt.axvspan(start / fs, end / fs, color='blue', alpha=0.3, label='Low Gamma')

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.xlim([0, time[-1]])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()

    # Print stats
    high_duration = sum([(end - start) / fs for start, end in high_blocks_s])
    low_duration = sum([(end - start) / fs for start, end in low_blocks_s])
    print(f"High gamma blocks: {len(high_blocks_s)} | Total duration: {high_duration:.2f} s")
    print(f"Low gamma blocks:  {len(low_blocks_s)} | Total duration: {low_duration:.2f} s")



def plot_param_spectra_high_low(
    model: SpectralTimeModel,
    high_windows: List[int],
    low_windows: List[int],
    window_times: List[Tuple[int, int]],
    title: str = "Average Parameterized Spectra High vs Low Gamma"
):
    """
    Plot the average parameterized spectra for high vs low gamma windows.

    Args:
        model : Fitted SpectralTimeModel.
        high_windows : Either indices or start sample values for high gamma windows.
        low_windows : Either indices or start sample values for low gamma windows.
        window_times : List of (start, end) sample indices for each window.
        title : Plot title.
    """
    # Extract spectrogram: shape (n_windows, n_freqs)
    spectra = model.spectrogram.T
    freqs = model.freqs

    # If user passed window start times, convert to indices
    win_starts = [start for start, _ in window_times]
    if any(w not in range(len(spectra)) for w in high_windows + low_windows):
        high_inds = [i for i, s in enumerate(win_starts) if s in high_windows]
        low_inds = [i for i, s in enumerate(win_starts) if s in low_windows]
    else:
        high_inds = high_windows
        low_inds = low_windows

    # Compute mean and std
    avg_high, std_high = np.nanmean(spectra[high_inds], axis=0), np.nanstd(spectra[high_inds], axis=0)
    avg_low, std_low = np.nanmean(spectra[low_inds], axis=0), np.nanstd(spectra[low_inds], axis=0)

    # Plot with shaded error
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, avg_high, color="red", label=f"High Gamma (n={len(high_inds)})")
    plt.fill_between(freqs, avg_high - std_high, avg_high + std_high, color="red", alpha=0.3)

    plt.plot(freqs, avg_low, color="blue", label=f"Low Gamma (n={len(low_inds)})")
    plt.fill_between(freqs, avg_low - std_low, avg_low + std_low, color="blue", alpha=0.3)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Log Power")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np





# Plotting for method comparison blocks

def plot_method_comparison_blocks(lfp_signal, fs, comp_high_blocks, comp_low_blocks, info_label="", title="Method Comparison: Gamma Blocks"):
    """
    Plot LFP with high/low gamma blocks derived from Welch vs Multitaper comparison.
    """
    n_high = len(comp_high_blocks)
    n_low  = len(comp_low_blocks)
    dur_high = sum(e - s for s, e in comp_high_blocks)
    dur_low  = sum(e - s for s, e in comp_low_blocks)

    time = np.arange(len(lfp_signal)) / fs
    plt.figure(figsize=(18, 6))
    plt.plot(time, lfp_signal, color="black", linewidth=0.7, label="LFP")

    # Plot high and low blocks
    for s, e in comp_high_blocks:
        plt.axvspan(s, e, facecolor="red", alpha=0.4, hatch="//", edgecolor="red")
    for s, e in comp_low_blocks:
        plt.axvspan(s, e, facecolor="blue", alpha=0.4, hatch="//", edgecolor="blue")

    legend_handles = [
        mpatches.Patch(facecolor="red", alpha=0.4, hatch="//", edgecolor="red", label="High Gamma (Method Comparison)"),
        mpatches.Patch(facecolor="blue", alpha=0.4, hatch="//", edgecolor="blue", label="Low Gamma (Method Comparison)")
    ]
    plt.legend(handles=legend_handles, loc="upper right")
    plt.xlabel("Time (s)")
    plt.ylabel("LFP (µV)")
    plt.title(f"{title} – {info_label}")
    plt.tight_layout()
    plt.show()

    print(f"[{info_label}]")
    print(f"High Gamma (Method Comparison): {n_high} blocks | Total duration: {dur_high:.2f} s")
    print(f"Low Gamma (Method Comparison):  {n_low} blocks | Total duration: {dur_low:.2f} s")



# Wrapper that does both

def get_and_plot_method_comparison_blocks(
    lfp_signal, fs,
    welch_high, welch_low, mt_high, mt_low,
    mode="overlap",
    plot=True
):
    """
    Compute and optionally plot gamma blocks from Welch vs Multitaper comparison.

    Returns
    -------
    comp_high_blocks, comp_low_blocks
    """
    comp_high, comp_low, info = compute_method_comparison_blocks(
        welch_high, welch_low, mt_high, mt_low, mode=mode
    )

    if plot:
        plot_method_comparison_blocks(lfp_signal, fs, comp_high, comp_low, info)

    return comp_high, comp_low
# ===========================================================

def plot_patch_lfp_aligned(patch_times, patch_signal, lfp_times, lfp_signal,
                           t_start_ms=80000, t_end_ms=154000, fs_common=60000):
    """
    Plot patch and LFP data aligned to a common time axis (in minutes) 
    with interpolation.

    Args:
        patch_times : array (ms) - time points for patch data
        patch_signal : array - patch data values
        lfp_times : array (ms) - time points for LFP data
        lfp_signal : array - LFP data values
        t_start_ms : start of common time axis (ms)
        t_end_ms : end of common time axis (ms)
        fs_common : sampling rate for interpolation (Hz, default 60000)
    """

    # Convert times to minutes
    patch_times_min = patch_times / 60000
    lfp_times_min = lfp_times / 60000

    # Define common time axis (minutes)
    common_time_min = np.arange(t_start_ms / 60000, t_end_ms / 60000, 1 / fs_common)

    # Interpolate both signals
    patch_interp = np.interp(common_time_min, patch_times_min, patch_signal)
    lfp_interp = np.interp(common_time_min, lfp_times_min, lfp_signal)

    # Plot Patch Data
    plt.figure(figsize=(12, 4))
    plt.plot(common_time_min, patch_interp, label='Patch Data', color='darkgreen')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Voltage')
    plt.title('Patch Data with Common Time Axis')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot LFP Data
    plt.figure(figsize=(12, 4))
    plt.plot(common_time_min, lfp_interp, label='LFP Data', color='black')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Voltage')
    plt.title('LFP Data with Common Time Axis')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return common_time_min, patch_interp, lfp_interp