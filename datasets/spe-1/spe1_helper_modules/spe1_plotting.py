import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Sequence, Optional, Tuple, Literal, List
from spikeparam_plotting import *
from specparam import SpectralTimeModel
import matplotlib.patches as mpatches


def plot_spike_time_histogram_with_windows(
    spk_times_ms: np.ndarray,
    window_times: List[Tuple[int, int]],   # (start_sample, end_sample)
    lfp_fs: float,
    bins: int = 300,
    mode: str = "lines",                   # "lines" | "rugs" | "bands"
    stride: int = 20,                      # plot every Nth window
    xlim_ms: Optional[Tuple[float, float]] = None,
    spike_color: str = "blue",
    window_color: str = "orange",
    alpha: float = 0.6,
    title: str = "Spike Time Distribution with Sliding Windows",
    save_path: Optional[str] = None,
):
    """
    Plot spike-time histogram with overlaid sliding-window markers.

    mode:
      - "lines": a thin vertical line at each window start (clean & fast)
      - "rugs": short ticks at bottom of the axis at each window start
      - "bands": translucent spans for [start, end] (can look like bars if many)
    stride:
      - draw every Nth window to avoid visual overload
    xlim_ms:
      - (start_ms, end_ms) to zoom in
    """
    # convert window sample indices -> ms
    win_ms = np.array([(s / lfp_fs * 1000.0, e / lfp_fs * 1000.0) for s, e in window_times])

    # optional zoom: also restrict spikes to the range we’ll show (helps speed)
    if xlim_ms is not None:
        x0, x1 = xlim_ms
        sel = (win_ms[:, 1] >= x0) & (win_ms[:, 0] <= x1)
        win_ms = win_ms[sel]
        spk_sel = (spk_times_ms >= x0) & (spk_times_ms <= x1)
        spk_plot = spk_times_ms[spk_sel]
    else:
        spk_plot = spk_times_ms

    plt.figure(figsize=(12, 6))
    n, edges, _ = plt.hist(
        spk_plot, bins=bins, color=spike_color, alpha=0.7, edgecolor="black", label="Spike times"
    )

    # draw windows with stride
    win_ms_strided = win_ms[::max(1, stride)]

    if mode == "lines":
        # thin line at each start
        for i, (s_ms, e_ms) in enumerate(win_ms_strided):
            plt.axvline(s_ms, color=window_color, linestyle='-', linewidth=0.6, alpha=alpha,
                        label="LFP window start" if i == 0 else "")
    elif mode == "rugs":
        # short ticks at bottom
        ymin, ymax = plt.ylim()
        rug_y = ymin + 0.02 * (ymax - ymin)
        for i, (s_ms, e_ms) in enumerate(win_ms_strided):
            plt.plot([s_ms, s_ms], [ymin, rug_y], color=window_color, alpha=alpha, linewidth=0.8,
                     label="LFP window start" if i == 0 else "")
        plt.ylim(ymin, ymax)  # restore
    elif mode == "bands":
        # translucent spans for the full window (use small alpha; can look like bars)
        for i, (s_ms, e_ms) in enumerate(win_ms_strided):
            plt.axvspan(s_ms, e_ms, color=window_color, alpha=0.15,
                        label="LFP window" if i == 0 else "")
    else:
        raise ValueError("mode must be one of: 'lines', 'rugs', 'bands'")

    if xlim_ms is not None:
        plt.xlim(*xlim_ms)

    # de-duplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.xlabel("Time (ms)")
    plt.ylabel("Spike count")
    plt.title(title + f"  (windows shown: {len(win_ms_strided)}/{len(win_ms)})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
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



def plot_gamma_blocks_generic(
    lfp_signal: np.ndarray,
    fs: float,
    high_blocks: List[Tuple[float, float]],
    low_blocks: List[Tuple[float, float]],
    title: str = "LFP with Gamma Blocks",
    info_label: Optional[str] = None,
    color_high: str = "red",
    color_low: str = "blue",
    hatch_high: Optional[str] = None,
    hatch_low: Optional[str] = None,
    time_range: Optional[Tuple[float, float]] = None,  # 
):
    """
    General-purpose plotting for gamma blocks (works for both single-method and method-comparison).

    Parameters
    ----------
    lfp_signal : np.ndarray
        1D LFP time series.
    fs : float
        Sampling frequency (Hz).
    high_blocks, low_blocks : list of (start, end) in seconds.
    time_range : tuple (start_sec, end_sec), optional
        If provided, zooms into this time window.
    """
    import matplotlib.patches as mpatches

    # --- Setup full time array ---
    time = np.arange(len(lfp_signal)) / fs

    # --- If time_range is given, zoom into that section ---
    if time_range is not None:
        t_start, t_end = time_range
        idx_start = int(t_start * fs)
        idx_end = int(t_end * fs)

        time = time[idx_start:idx_end]
        lfp_signal = lfp_signal[idx_start:idx_end]

        # Crop block lists to time_range
        def crop_blocks(blocks):
            return [(max(s, t_start), min(e, t_end)) for s, e in blocks if e > t_start and s < t_end]

        high_blocks = crop_blocks(high_blocks)
        low_blocks = crop_blocks(low_blocks)

    # --- Plot ---
    plt.figure(figsize=(16, 5))
    plt.plot(time, lfp_signal, color="black", linewidth=0.8, label="LFP Signal")

    for s, e in high_blocks:
        plt.axvspan(s, e, facecolor=color_high, alpha=0.3,
                    edgecolor=color_high, hatch=hatch_high)
    for s, e in low_blocks:
        plt.axvspan(s, e, facecolor=color_low, alpha=0.3,
                    edgecolor=color_low, hatch=hatch_low)

    # --- Legend ---
    legend_handles = [
        mpatches.Patch(facecolor=color_high, alpha=0.3, hatch=hatch_high or '',
                       edgecolor=color_high, label="High Gamma"),
        mpatches.Patch(facecolor=color_low, alpha=0.3, hatch=hatch_low or '',
                       edgecolor=color_low, label="Low Gamma")
    ]
    plt.legend(handles=legend_handles, loc="upper right")

    # --- Title ---
    label_text = f"{title}" + (f" – {info_label}" if info_label else "")
    plt.title(label_text)
    plt.xlabel("Time (s)")
    plt.ylabel("LFP (µV)")
    plt.tight_layout()
    plt.show()

    # --- Stats ---
    print(f"{info_label if info_label else '[Blocks]'}")
    print(f"High Gamma: {len(high_blocks)} blocks | Total duration: {sum(e - s for s, e in high_blocks):.2f} s")
    print(f"Low Gamma:  {len(low_blocks)} blocks | Total duration: {sum(e - s for s, e in low_blocks):.2f} s")






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


def plot_spike_groups_from_labels(sp, spike_df_labeled, label_column, label_groups, group_names):
    """
    Plot average waveforms for given label groups from a labeled spike dataframe,
    and print the number of spikes in each group.

    Parameters
    ----------
    sp : Spike
        Spike object (with .plot method).
    spike_df_labeled : pd.DataFrame
        DataFrame containing spike labels.
    label_column : str
        Column in spike_df_labeled containing labels (e.g., 'method_block_label').
    label_groups : list of str
        Labels to include in plotting (in order).
    group_names : list of str
        Names to display for the groups in the legend.
    """
    ind_groups = []
    
    for lbl, name in zip(label_groups, group_names):
        inds = spike_df_labeled.loc[spike_df_labeled[label_column] == lbl, 'spk_id'].values
        valid_inds = [i for i in inds if i < len(sp.spikes)]
        ind_groups.append(valid_inds)
        print(f"Group '{name}' → {len(valid_inds)} spikes")

    # Plot with Spike class built-in plotting
    sp.plot(
        groups=True,
        ind_groups=ind_groups,
        group_names=group_names,
        plot_average=True,
        plot_average_std=True
    )



def visualize_flat_vs_steep_sanity(
    spike_df: pd.DataFrame,
    summary_df_multitaper: pd.DataFrame,
    flattest_windows: pd.DataFrame,
    steepest_windows: pd.DataFrame,
    window_times_mt: List[Tuple[int, int]],
    lfp_signal: np.ndarray,
    lfp_fs: float,
    time_range: Optional[Tuple[float, float]] = None,
    # viz styling
    color_flat: str = "#009E73",   # teal
    color_steep: str = "#CC79A7",  # magenta
    hatch_flat: Optional[str] = "//",
    hatch_steep: Optional[str] = "\\\\",
    plot_waveforms: bool = False,
    sp: Optional[object] = None,
    # NEW: which column already holds labels
    label_column: str = "method_block_label",
) -> pd.DataFrame:
    """
    Sanity-check visualization for flat vs steep windows.

    NOTE: This version does NOT (re)label spikes.
          It expects `spike_df[label_column]` to already exist.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- require precomputed labels ---
    if label_column not in spike_df.columns:
        raise ValueError(
            f"Expected pre-labeled spikes: column '{label_column}' not found in spike_df."
        )

    # blocks in seconds (for shading & hist bands only)
    blocks_flat_sec  = [(window_times_mt[i][0]/lfp_fs, window_times_mt[i][1]/lfp_fs)
                        for i in flattest_windows.index]
    blocks_steep_sec = [(window_times_mt[i][0]/lfp_fs, window_times_mt[i][1]/lfp_fs)
                        for i in steepest_windows.index]

    

    # (1) overlay on LFP (reuses your plotter)
    plot_gamma_blocks_generic(
        lfp_signal=lfp_signal,
        fs=lfp_fs,
        high_blocks=blocks_steep_sec,    # steep
        low_blocks=blocks_flat_sec,      # flat
        title="Sanity: STEEP vs FLAT windows",
        info_label=None,
        color_high=color_steep,
        color_low=color_flat,
        hatch_high=hatch_steep,
        hatch_low=hatch_flat,
        time_range=time_range
    )

    # (2) spike histogram + FLAT bands
    flat_window_times  = [window_times_mt[i] for i in flattest_windows.index]
    plot_spike_time_histogram_with_windows(
        spk_times_ms=spike_df["spk_times_ms"].to_numpy(),
        window_times=flat_window_times,
        lfp_fs=lfp_fs,
        mode="bands", stride=1,
        window_color=color_flat, alpha=0.15,
        title="Spikes with FLAT windows"
    )

    # (3) spike histogram + STEEP bands
    steep_window_times = [window_times_mt[i] for i in steepest_windows.index]
    plot_spike_time_histogram_with_windows(
        spk_times_ms=spike_df["spk_times_ms"].to_numpy(),
        window_times=steep_window_times,
        lfp_fs=lfp_fs,
        mode="bands", stride=1,
        window_color=color_steep, alpha=0.15,
        title="Spikes with STEEP windows"
    )

    # (4) counts bar from existing labels
    counts = spike_df[label_column].value_counts()
    order = [lbl for lbl in ["flat","steep","none"] if lbl in counts.index]
    color_map = {"flat": color_flat, "steep": color_steep}
    colors = [color_map.get(lbl, "#9E9E9E") for lbl in order]  # fixed small typo

    plt.figure(figsize=(5,3))
    plt.bar(order, counts[order].values, color=colors)
    plt.title("Spike labels (precomputed)"); plt.xlabel("label"); plt.ylabel("n spikes")
    plt.tight_layout(); plt.show()

  

    # return unchanged df (just for symmetry with old API)
    return spike_df





def plot_param_spectra_two_groups(
    model,
    group_a: Sequence,               # indices OR [(start_sec, end_sec), ...]
    group_b: Sequence,               # indices OR [(start_sec, end_sec), ...]
    window_times: List[Tuple[int,int]],
    *,
    label_a: str = "Group A",
    label_b: str = "Group B",
    component: Literal["full","aperiodic","peak"] = "full",
    space: Literal["log","linear"] = "log",
    freq_range: Optional[Tuple[float,float]] = None,
    ci: Literal["sd","sem", None] = "sd",
    color_a: str = "#009E73",
    color_b: str = "#CC79A7",
    title: Optional[str] = None,
    lfp_fs: Optional[float] = None,  # required iff groups are blocks in seconds
):
    """
    Plot average Specparam *modeled* spectra for two groups.

    `group_a` / `group_b` may be:
      - iterable of window indices (ints), OR
      - iterable of (start_sec, end_sec) blocks (then pass `lfp_fs`)

    Uses per-window Specparam models: model.get_model(i).get_data(component, space)
    """

    freqs = np.asarray(model.freqs)
    nwin  = int(model.n_time_windows)

    # --- map group (indices or seconds-blocks) -> window indices
    starts = np.array([s for s, _ in window_times], dtype=int)
    start_to_idx = {int(s): i for i, s in enumerate(starts)}

    def to_indices(group):
        if len(group) == 0:
            return np.array([], dtype=int)
        first = group[0]
        # seconds-blocks
        if isinstance(first, (tuple, list)) and len(first) == 2 and not isinstance(first, (int, np.integer)):
            if lfp_fs is None:
                raise ValueError("lfp_fs is required when groups are blocks in seconds.")
            idxs = []
            for (s_sec, _e_sec) in group:
                s_idx = int(round(float(s_sec) * float(lfp_fs)))
                if s_idx in start_to_idx:
                    idxs.append(start_to_idx[s_idx])
            return np.unique(np.array(idxs, dtype=int))
        # indices
        return np.unique(np.array(group, dtype=int))

    idx_a = to_indices(group_a)
    idx_b = to_indices(group_b)

    # keep only valid
    idx_a = idx_a[(idx_a >= 0) & (idx_a < nwin)]
    idx_b = idx_b[(idx_b >= 0) & (idx_b < nwin)]

    if len(idx_a) == 0 or len(idx_b) == 0:
        print(f"[plot] nothing to plot: {label_a} n={len(idx_a)}, {label_b} n={len(idx_b)}")
        return

    # --- collect modeled spectra per window (rows = windows)
    def stack_group(indices):
        rows = []
        for i in indices:
            m = model.get_model(int(i))
            if m is None:
                continue
            y = m.get_data(component=component, space=space)
            if y is None:
                continue
            rows.append(np.asarray(y))
        return np.vstack(rows) if rows else np.empty((0, len(freqs)))

    A = stack_group(idx_a)   # shape: (nA, n_freqs)
    B = stack_group(idx_b)   # shape: (nB, n_freqs)

    if A.shape[0] == 0 or B.shape[0] == 0:
        print(f"[plot] nothing to plot after stacking: {label_a} n={A.shape[0]}, {label_b} n={B.shape[0]}")
        return

    # optional freq crop
    if freq_range is not None:
        lo, hi = freq_range
        sel = (freqs >= lo) & (freqs <= hi)
        if not np.any(sel):
            print("[plot] freq_range produced empty selection.")
            return
        freqs = freqs[sel]
        A, B = A[:, sel], B[:, sel]

    # averages & error
    mean_a = np.nanmean(A, axis=0)
    mean_b = np.nanmean(B, axis=0)

    err_a = err_b = None
    if ci == "sd":
        err_a, err_b = np.nanstd(A, axis=0), np.nanstd(B, axis=0)
    elif ci == "sem":
        err_a = np.nanstd(A, axis=0) / np.sqrt(max(A.shape[0], 1))
        err_b = np.nanstd(B, axis=0) / np.sqrt(max(B.shape[0], 1))
    elif ci is not None:
        raise ValueError("ci must be 'sd', 'sem', or None")

    # --- plot
    plt.figure(figsize=(10,5))
    plt.plot(freqs, mean_a, color=color_a, label=f"{label_a} (n={A.shape[0]})")
    if err_a is not None:
        plt.fill_between(freqs, mean_a - err_a, mean_a + err_a, color=color_a, alpha=0.25)

    plt.plot(freqs, mean_b, color=color_b, label=f"{label_b} (n={B.shape[0]})")
    if err_b is not None:
        plt.fill_between(freqs, mean_b - err_b, mean_b + err_b, color=color_b, alpha=0.25)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(("Log Power" if space == "log" else "Power") + f" — {component}")
    plt.title(title or f"Average Parameterized Spectra ({component}, {space})")
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_spectra_and_auc(
    model,
    win_idx: int,
    band: Tuple[float, float] = (30.0, 90.0),
    freq_range: Optional[Tuple[float, float]] = None,  # display zoom only
    title: Optional[str] = None,
):
    """
    Plot full vs aperiodic *log* spectra for one window, and annotate AUC computed from
    (full_linear − aperiodic_linear) within `band` using trapezoid rule.
    """
    m = model.get_model(int(win_idx))
    freqs = np.asarray(model.freqs)
    sel = (freqs >= band[0]) & (freqs <= band[1])

    full = m.get_model(component="full",      space="log")
    ap   = m.get_model(component="aperiodic", space="log")
    

    resid = full - ap  # aperiodic-adjusted spectrum; can be negative 

    

    auc = np.trapz(resid[sel], freqs[sel])

    # --- plot: full vs aperiodic in LOG space, band highlighted, AUC annotated ---
    plt.figure(figsize=(9, 4.8))
    plt.plot(freqs, full,  lw=1.8, color="#333333", label="Full (log)")
    plt.plot(freqs, ap,    lw=1.8, color="#ff7f0e", label="Aperiodic (log)")
    plt.axvspan(band[0], band[1], color="#4c72b0", alpha=0.12, lw=0)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Log Power")
    plt.title(title or f"Window {win_idx}: full vs aperiodic (log); AUC in {int(band[0])}–{int(band[1])} Hz")
    plt.legend(loc="best")

    # neat AUC annotation
    plt.text(
        0.98, 0.04,
        f"AUC (linear residual) = {auc:.4g}\nBand = {int(band[0])}–{int(band[1])} Hz",
        transform=plt.gca().transAxes,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, lw=0.5)
    )

    plt.tight_layout()
    plt.show()

    return auc 
