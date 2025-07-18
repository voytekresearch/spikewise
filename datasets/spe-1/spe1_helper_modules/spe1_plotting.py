import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from spikeparam_plotting import *
from specparam import SpectralTimeModel


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

def plot_gamma_blocks_merged(high_blocks, low_blocks, window_len_sec, total_duration_sec):
    """
    Plot merged high/low gamma blocks with legend showing block count and total duration.
    """
    fig, ax = plt.subplots(figsize=(12, 2))

    for start, end in high_blocks:
        ax.axvspan(start * window_len_sec, (end + 1) * window_len_sec, color='red', alpha=0.3)

    for start, end in low_blocks:
        ax.axvspan(start * window_len_sec, (end + 1) * window_len_sec, color='blue', alpha=0.3)

    ax.set_xlim(0, total_duration_sec)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("Gamma Power Blocks")
    ax.legend([
        f"High gamma (n={len(high_blocks)}, {compute_block_lengths(high_blocks, window_len_sec):.1f}s)",
        f"Low gamma (n={len(low_blocks)}, {compute_block_lengths(low_blocks, window_len_sec):.1f}s)"
    ])
    plt.show()
