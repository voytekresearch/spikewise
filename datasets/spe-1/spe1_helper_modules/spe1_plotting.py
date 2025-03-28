import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from spikeparam_plotting import *


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
        show_sig=show_sig
    )
