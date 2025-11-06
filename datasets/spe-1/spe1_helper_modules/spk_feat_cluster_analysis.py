"""
skp_feat_cluster_analysis.py
----------------------------
Analysis utilities for identifying and visualizing spike feature clustering patterns
and their temporal dynamics.

This module provides:
  • Automated detection of multimodal (clustered) spike feature distributions.
  • Lightweight, dependency-free 1D K-means clustering for separating spike features
    into groups (e.g., "low" / "high").
  • Visualizations of cluster evolution over recording time.
  • ISI-based transition analysis and detection of "burst onsets" defined as
    long-ISI (high) spikes followed by runs of short-ISI (low) spikes.

Assumptions
-----------
• Spike timestamps in the dataset are stored in **milliseconds** (ms).
• All plotting functions display time on the x-axis in **seconds** (s).
• Any `time_range` parameters are specified in **seconds**.

Typical workflow
----------------
1) Detect and cluster features that look multimodal:
     df_clustered, report = cluster_multimodal_features(df, max_k=3)
2) (Optional) Visualize cluster presence over time:
     plot_clusters_over_time_min(df_clustered, "spk_times_ms", "log_isi_cluster", time_unit="ms")
3) Inspect ISI cluster transitions:
     trans = isi_transition_matrix(df_clustered, "log_isi_cluster")
4) Identify and plot burst onsets (long→short ISI):
     df_marked = mark_high_burst_onsets(df_clustered, "log_isi_cluster", "spk_times_ms", min_low_run=2)
     plot_long_to_short_isi_onsets(df_marked, "spk_times_ms", "log_isi_cluster", time_range=(200, 400))
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import mne
from specparam import SpectralTimeModel
from typing import Optional, List, Tuple, Dict, Any, Literal
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, probplot

# ------------------------------------------------------------------------------------------- #
# ------------------------------ Cluster features that show grouped data --------------------- #
# ------------------------------------------------------------------------------------------- #

def kmeans_1d_cluster(
    df: pd.DataFrame,
    feature: str,
    k: int = 2,
    labels: Optional[List[str]] = None,
    new_col: Optional[str] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    plot: bool = False,
    bins: int = 40,
):
    """
    Perform simple 1D K-means clustering on a given numeric feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the feature to cluster.
    feature : str
        Column name of the feature to cluster.
    k : int, default=2
        Number of clusters (>= 2).
    labels : list of str, optional
        Custom cluster labels (length must equal k). Defaults to ["c0", ..., f"c{k-1}"].
    new_col : str, optional
        Name for the new cluster label column. Defaults to f"{feature}_cluster".
    max_iter : int, default=100
        Maximum number of Lloyd iterations.
    tol : float, default=1e-6
        Convergence threshold for center updates.
    plot : bool, default=False
        If True, plots a histogram colored by cluster with vertical cutoff lines.
    bins : int, default=40
        Number of histogram bins when `plot=True`.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with the new cluster label column added.
    centers : np.ndarray
        Cluster centers (sorted ascending).
    cutoffs : np.ndarray
        Midpoints between adjacent sorted centers (length k-1).

    Notes
    -----
    • Non-finite values (NaN/±inf) are ignored for fitting and remain NaN in the label column.
    • Initialization uses quantiles for stability; falls back to linear spacing if needed.
    """
    if feature not in df.columns:
        raise KeyError(f"Column '{feature}' not found in DataFrame.")
    if k < 2:
        raise ValueError("k must be >= 2")

    df_out = df.copy()
    x = pd.to_numeric(df_out[feature], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x)
    idx_valid = np.where(valid)[0]
    data = x[valid]

    if data.size < 2 or np.allclose(data, data[0], rtol=0, atol=0):
        out_col = new_col if new_col else f"{feature}_cluster"
        df_out[out_col] = np.nan
        return df_out, np.array([]), np.array([])

    if labels is None:
        labels = [f"c{i}" for i in range(k)]
    if len(labels) != k:
        raise ValueError(f"`labels` length ({len(labels)}) must equal k ({k}).")
    out_col = new_col if new_col else f"{feature}_cluster"

    # Quantile init (robust)
    qs = np.linspace(0, 1, k + 2)[1:-1]
    centers = np.quantile(data, qs)
    if np.unique(centers).size < k:
        mn, mx = np.min(data), np.max(data)
        if np.isclose(mn, mx):
            df_out[out_col] = np.nan
            return df_out, np.array([]), np.array([])
        centers = np.linspace(mn, mx, k)

    # Lloyd’s algorithm
    for _ in range(max_iter):
        dists = np.abs(data[:, None] - centers[None, :])
        assign = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        changed = False
        for i in range(k):
            mask_i = (assign == i)
            if np.any(mask_i):
                c_new = data[mask_i].mean()
                if not np.isclose(c_new, centers[i], atol=tol, rtol=0):
                    new_centers[i] = c_new
                    changed = True
        centers = new_centers
        if not changed:
            break

    # Sort centers + remap labels
    order = np.argsort(centers)
    centers = centers[order]
    remap = {old_i: new_i for new_i, old_i in enumerate(order)}
    assign_sorted = np.vectorize(remap.get)(assign)
    cutoffs = (centers[:-1] + centers[1:]) / 2.0

    # Write labels back at original indices
    label_array = np.full(x.shape, np.nan, dtype=object)
    for i in range(k):
        full_idx = idx_valid[assign_sorted == i]
        label_array[full_idx] = labels[i]
    df_out[out_col] = label_array

    if plot:
        plt.figure(figsize=(7.5, 4.2))
        for i in range(k):
            plt.hist(data[assign_sorted == i], bins=bins, alpha=0.6, label=labels[i])
        for c in cutoffs:
            plt.axvline(c, linestyle="--", linewidth=2)
        plt.xlabel(feature)
        plt.ylabel("count")
        plt.title(f"{feature} clusters (k={k})")
        if k <= 10:
            plt.legend()
        plt.tight_layout()
        plt.show()

    return df_out, centers, cutoffs


def _fd_bins(data: np.ndarray) -> int:
    """
    Freedman–Diaconis rule for choosing a reasonable histogram bin count.

    Parameters
    ----------
    data : np.ndarray
        1D numeric array.

    Returns
    -------
    int
        Number of bins (clipped to [10, 80]).
    """
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return min(50, max(10, int(np.sqrt(data.size))))
    bin_width = 2 * iqr * (data.size ** (-1/3))
    if bin_width <= 0:
        return min(50, max(10, int(np.sqrt(data.size))))
    bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return int(np.clip(bins, 10, 80))


def _count_peaks_smooth(hist: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Count local peaks in a smoothed histogram.

    Smoothing kernel reduces spurious wiggles before peak counting.

    Parameters
    ----------
    hist : np.ndarray
        Histogram counts.

    Returns
    -------
    n_peaks : int
        Number of peaks.
    peak_indices : np.ndarray
        Indices of detected peaks in the smoothed histogram.
    """
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    sm = np.convolve(hist, kernel, mode='same')
    peaks = np.where((sm[1:-1] > sm[:-2]) & (sm[1:-1] > sm[2:]))[0] + 1
    return len(peaks), peaks


def _looks_clustered_1d(
    data: np.ndarray,
    min_peak_ratio: float = 0.10,
    max_valley_ratio: float = 0.70,
    max_k: int = 3
) -> Tuple[int, Dict[str, Any]]:
    """
    Detect potential multimodality (e.g., 2–3 distinct modes) in 1D data
    using a smoothed histogram peak analysis.

    Parameters
    ----------
    data : np.ndarray
        1D numeric data.
    min_peak_ratio : float, default=0.10
        Minimum relative height for the secondary peak (vs. tallest peak).
    max_valley_ratio : float, default=0.70
        Maximum permitted valley depth ratio between the two tallest peaks.
        Lower values indicate better separation.
    max_k : int, default=3
        Maximum clusters suggested (1..3).

    Returns
    -------
    k_suggest : int
        Suggested number of clusters (1, 2, or 3).
    diagnostics : dict
        Histogram/peaks diagnostics, including `valley_ratio` and approximate peak locations.

    Notes
    -----
    • This is a fast heuristic; it won’t replace formal tests (e.g., Hartigan’s dip test).
    """
    if data.size < 50:
        # Small samples can be noisy; leave thresholds as-is but keep this in mind.
        pass
    bins = _fd_bins(data)
    hist, edges = np.histogram(data, bins=bins)
    if hist.max() == 0:
        return 1, {"reason": "empty hist"}

    h = hist.astype(float) / hist.max()
    n_peaks, peak_idx = _count_peaks_smooth(h)
    if n_peaks < 2:
        return 1, {"peaks": n_peaks}

    # Two tallest peaks
    peak_heights = h[peak_idx]
    top2 = np.argsort(peak_heights)[-2:]
    p_idx = peak_idx[top2]
    p_idx.sort()
    p1, p2 = p_idx[0], p_idx[1]
    h1, h2 = h[p1], h[p2]

    if min(h1, h2) < min_peak_ratio:
        return 1, {"reason": "secondary peak too small", "h1": h1, "h2": h2}

    valley = h[(p1+1):p2].min() if p2 > p1 + 1 else min(h1, h2)
    valley_ratio = valley / min(h1, h2) if min(h1, h2) > 0 else 1.0
    if valley_ratio >= max_valley_ratio:
        return 1, {"reason": "valley too shallow", "valley_ratio": valley_ratio}

    # At least bimodal; allow tri-modal if a third peak looks meaningful
    k_suggest = 2
    if max_k >= 3 and n_peaks >= 3:
        third_idx = [i for i in peak_idx if i not in (p1, p2)]
        third_heights = [h[i] for i in third_idx]
        if len(third_heights) > 0 and max(third_heights) >= min_peak_ratio:
            k_suggest = 3

    centers_approx = (edges[:-1] + edges[1:]) / 2.0
    approx_peaks_vals = centers_approx[peak_idx]
    return k_suggest, {
        "peaks": int(n_peaks),
        "peak_bins": peak_idx,
        "peak_vals_approx": approx_peaks_vals.tolist(),
        "valley_ratio": float(valley_ratio)
    }


def cluster_multimodal_features(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    max_k: int = 3,
    labels_map: Optional[Dict[str, List[str]]] = None,
    suffix: str = "_cluster",
    plot_each: bool = False,
    peak_ratio: float = 0.10,
    valley_ratio: float = 0.70,
    unique_min: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Detect features with clustered (multi-peak) distributions and cluster them via 1D K-means.

    For each numeric feature in `features` (or all numeric columns if None):
      • Detects multimodality via smoothed histogram peaks.
      • If clustered, applies 1D K-means (k=2 or 3 capped by `max_k`).
      • Adds a new label column named f"{feature}{suffix}".

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with spike features.
    features : list of str, optional
        Which columns to consider; defaults to all numeric columns.
    max_k : int, default=3
        Maximum clusters to consider (suggested: 2 or 3).
    labels_map : dict[str, list[str]], optional
        Optional mapping {feature: [labels]} to override default labels.
    suffix : str, default="_cluster"
        Suffix used to name the new label columns.
    plot_each : bool, default=False
        If True, show per-feature histogram plots after clustering.
    peak_ratio : float, default=0.10
        Minimum acceptable relative height for secondary peaks.
    valley_ratio : float, default=0.70
        Maximum acceptable valley/peak ratio between the two tallest peaks.
    unique_min : int, default=8
        Skip features with fewer than this many unique valid values.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with new cluster label columns for detected features.
    report : dict
        Per-feature metadata: `k`, `centers`, `cutoffs`, `diagnostics`, label `column`, and `labels`.

    Notes
    -----
    • Rows with non-finite feature values are ignored for fitting and remain NaN in the label column.
    • If labels_map[feature] length mismatches the chosen k, generic labels are used.
    """
    df_out = df.copy()

    # choose features
    if features is None:
        features = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    report: Dict[str, Dict[str, Any]] = {}

    for feat in features:
        x = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x)
        data = x[valid]
        if np.unique(data).size < unique_min:
            continue  # too few unique values to meaningfully cluster

        # detect modality
        k_suggest, diag = _looks_clustered_1d(
            data, min_peak_ratio=peak_ratio, max_valley_ratio=valley_ratio, max_k=max_k
        )
        if k_suggest < 2:
            continue  # unimodal; skip

        # decide labels
        if labels_map and feat in labels_map:
            labels = labels_map[feat]
            if len(labels) != k_suggest:
                labels = (['low', 'high'] if k_suggest == 2 else ['low', 'mid', 'high'][:k_suggest])
        else:
            labels = ['low', 'high'] if k_suggest == 2 else ['low', 'mid', 'high'][:k_suggest]

        # cluster
        new_col = f"{feat}{suffix}"
        df_out, centers, cutoffs = kmeans_1d_cluster(
            df_out, feature=feat, k=k_suggest, labels=labels, new_col=new_col, plot=plot_each
        )

        report[feat] = {
            "k": int(k_suggest),
            "centers": centers,
            "cutoffs": cutoffs,
            "diagnostics": diag,
            "column": new_col,
            "labels": labels,
        }

    return df_out, report


def plot_clusters_over_time_min(
    df: pd.DataFrame,
    time_col: str,
    label_col: str,
    time_unit: str = "ms",
    fs: Optional[float] = None,
    bin_size_ms: float = 1000,
    sigma_bins: Optional[float] = None
):
    """
    Plot temporal evolution of cluster membership across the recording.

    Top panel: raster of spike times colored by cluster.
    Bottom panel: cluster-specific firing rates (binned, optionally smoothed).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns for time and cluster labels.
    time_col : str
        Spike time column; units depend on `time_unit`.
    label_col : str
        Cluster label column (e.g., "low", "high").
    time_unit : {'ms', 'samples'}, default='ms'
        Units of `time_col`. If 'samples', provide `fs`.
    fs : float, optional
        Sampling frequency in Hz when `time_unit='samples'`.
    bin_size_ms : float, default=1000
        Bin width for rate calculation (milliseconds).
    sigma_bins : float, optional
        Gaussian smoothing sigma in units of bins (None = no smoothing).

    Notes
    -----
    • Spike times are converted to seconds for plotting/rate computation.
    • Non-finite times or NaN labels are ignored.
    """
    # ---- extract
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
    lab = df[label_col].astype(object).to_numpy()
    valid = np.isfinite(t) & pd.notna(lab)
    if not valid.any():
        print("No valid points.")
        return

    # seconds
    if time_unit == "ms":
        t_sec = t[valid] / 1000.0
    elif time_unit == "samples":
        if not fs or fs <= 0:
            raise ValueError("Provide fs when time_unit='samples'.")
        t_sec = t[valid] / fs
    else:
        raise ValueError("time_unit must be 'ms' or 'samples'.")

    labels = lab[valid].astype(object)
    clusters = [str(x) for x in pd.unique(labels)]

    # bins
    bin_w = bin_size_ms / 1000.0
    t0, t1 = float(np.min(t_sec)), float(np.max(t_sec))
    if np.isclose(t0, t1):
        t1 = t0 + bin_w
    edges = np.arange(t0, t1 + bin_w, bin_w)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _smooth(x, s):
        if not s or s <= 0:
            return x
        half = int(np.ceil(4 * s))
        g = np.arange(-half, half + 1, dtype=float)
        k = np.exp(-0.5 * (g / s) ** 2)
        k /= k.sum()
        return np.convolve(x, k, mode="same")

    fig, (ax_raster, ax_rate) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1.2]}
    )

    # raster
    for i, cl in enumerate(clusters):
        m = (labels == cl)
        if m.sum() == 0:
            continue
        y = np.full(m.sum(), i + 1, float) + (np.random.rand(m.sum()) - 0.5) * 0.03
        ax_raster.scatter(t_sec[m], y, s=4, alpha=0.8, label=str(cl))
    ax_raster.set_yticks(range(1, len(clusters) + 1))
    ax_raster.set_yticklabels(clusters)
    ax_raster.set_ylabel("cluster (raster)")
    ax_raster.legend(loc="upper right", fontsize=8)

    # rates
    for cl in clusters:
        m = (labels == cl)
        counts, _ = np.histogram(t_sec[m], bins=edges)
        rate = _smooth(counts / bin_w, sigma_bins)
        ax_rate.plot(centers, rate, label=str(cl))
    ax_rate.set_ylabel("rate (spikes/s)")
    ax_rate.set_xlabel("time (s)")
    ax_rate.legend(loc="upper right", fontsize=9)
    ax_rate.set_xlim(edges[0], edges[-1])

    ttl = f"ISI clusters over time (bin={bin_size_ms} ms"
    if sigma_bins:
        ttl += f", σ={sigma_bins} bins"
    ttl += ")"
    fig.suptitle(ttl, y=0.98)
    plt.tight_layout()
    plt.show()


def isi_transition_matrix(df: pd.DataFrame, cluster_col: str = "log_isi_cluster") -> pd.DataFrame:
    """
    Compute the one-step transition probability matrix between ISI cluster labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the sequential ISI cluster labels.
    cluster_col : str, default='log_isi_cluster'
        Column with categorical labels (e.g., 'low', 'high').

    Returns
    -------
    pd.DataFrame
        Square transition probability matrix:
        rows = current label, columns = next label.
    """
    seq = df[cluster_col].dropna().astype(str).to_numpy()
    clusters = sorted(np.unique(seq))
    n = len(clusters)
    mat = np.zeros((n, n))
    for a, b in zip(seq[:-1], seq[1:]):
        i, j = clusters.index(a), clusters.index(b)
        mat[i, j] += 1
    mat = mat / mat.sum(axis=1, keepdims=True)
    return pd.DataFrame(mat, index=clusters, columns=clusters)


def mark_high_burst_onsets(
    df: pd.DataFrame,
    cluster_col: str = "log_isi_cluster",
    time_col: str = "spk_times_ms",
    min_low_run: int = 1,
) -> pd.DataFrame:
    """
    Identify long-ISI (high) spikes that start a burst of short-ISI (low) spikes.

    A spike at index i is marked as an onset if:
      • df[cluster_col].iloc[i] == "high", and
      • the next `min_low_run` spikes are all labeled "low".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least `cluster_col` and `time_col`.
    cluster_col : str, default='log_isi_cluster'
        ISI cluster label column; must include "high" and "low".
    time_col : str, default='spk_times_ms'
        Spike time column (assumed **milliseconds**; only used for sorting).
    min_low_run : int, default=1
        Minimum number of consecutive "low" spikes required after a "high" to flag an onset.

    Returns
    -------
    pd.DataFrame
        Copy of df with boolean column 'is_long_to_short_isi_onset' marking onset spikes.

    Notes
    -----
    • The function sorts by `time_col` before evaluation to ensure temporal order.
    • It does not enforce any absolute time-limit between spikes; only label sequence matters.
    """
    out = df.copy().sort_values(time_col).reset_index(drop=True)
    labs = out[cluster_col].astype(object).to_numpy()

    n = len(labs)
    is_low = labs == "low"
    next_low_all = np.zeros(n, dtype=bool)

    if n > min_low_run:
        for i in range(n - min_low_run):
            if labs[i] == "high" and np.all(is_low[i + 1: i + 1 + min_low_run]):
                next_low_all[i] = True

    out["is_long_to_short_isi_onset"] = next_low_all
    return out


def plot_long_to_short_isi_onsets(
    df_marked: pd.DataFrame,
    time_col: str = "spk_times_ms",          # always in milliseconds
    cluster_col: str = "log_isi_cluster",
    time_range: Optional[Tuple[float, float]] = None,  # in seconds
    show_counts: bool = True,
):
    """
    Visualize long→short ISI burst onsets over time.

    All spikes are plotted along a horizontal line (x-axis in seconds), colored by
    their ISI cluster. Spikes that mark the onset of a short-ISI burst (i.e., long-ISI
    spikes immediately followed by a run of `min_low_run` short-ISI spikes) are circled.

    Parameters
    ----------
    df_marked : pd.DataFrame
        DataFrame that contains a boolean column 'is_long_to_short_isi_onset' (from
        `mark_high_burst_onsets`), plus the time and cluster columns.
    time_col : str, default='spk_times_ms'
        Spike time column in **milliseconds**. It is converted to seconds for plotting.
    cluster_col : str, default='log_isi_cluster'
        Cluster label column containing 'low' and 'high'.
    time_range : tuple(float, float), optional
        (t_min, t_max) time window in **seconds** to display. If None, uses the full range.
    show_counts : bool, default=True
        If True, appends spike and onset counts for the plotted window in the title.

    Notes
    -----
    • Non-finite times or NaN labels are ignored.
    • If no spikes/onsets fall in the requested window, the x-axis is still set to that
      window and a message is shown on the plot.
    """
    color_map = {"low": "#1f77b4", "high": "#ff7f0e"}

    # convert from ms → s for plotting
    t_sec_all = pd.to_numeric(df_marked[time_col], errors="coerce").to_numpy(float) / 1000.0
    valid = np.isfinite(t_sec_all) & df_marked[cluster_col].notna()
    t_sec_all = t_sec_all[valid]
    labels_all = df_marked.loc[valid, cluster_col].astype(str).to_numpy()
    colors_all = np.array([color_map.get(lbl, "gray") for lbl in labels_all], dtype=object)

    # Onset times in seconds (LONG→SHORT)
    onset_mask_all = valid & (df_marked["is_long_to_short_isi_onset"] == True)
    t_on_sec_all = pd.to_numeric(df_marked.loc[onset_mask_all, time_col], errors="coerce").to_numpy(float) / 1000.0

    # Apply window in seconds
    if time_range is not None:
        tmin, tmax = map(float, time_range)
        in_window = (t_sec_all >= tmin) & (t_sec_all <= tmax)
        t_plot = t_sec_all[in_window]
        colors = colors_all[in_window]

        in_window_on = (t_on_sec_all >= tmin) & (t_on_sec_all <= tmax)
        t_on = t_on_sec_all[in_window_on]
    else:
        t_plot, colors, t_on = t_sec_all, colors_all, t_on_sec_all
        tmin, tmax = (t_sec_all.min(), t_sec_all.max()) if t_sec_all.size else (0.0, 1.0)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 2.6))
    if t_plot.size:
        ax.scatter(t_plot, np.ones_like(t_plot), s=6, c=colors, alpha=0.85)
    if t_on.size:
        ax.scatter(t_on, np.ones_like(t_on), s=60,
                   facecolors="none", edgecolors="black", linewidths=1.4,
                   label="LONG→SHORT onset (LONG spike)")

    ax.set_xlim(tmin, tmax)
    ax.set_xlabel("time (s)")
    ax.set_yticks([])

    title = f"Burst onsets: LONG→SHORT ISI ({tmin:.0f}–{tmax:.0f} s)"
    if show_counts:
        title += f"   [spikes: {t_plot.size}, onsets: {t_on.size}]"
    ax.set_title(title)

    if t_plot.size == 0 and t_on.size == 0:
        ax.text(
            0.5, 0.5, "No spikes / onsets in selected window",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="gray", alpha=0.8
        )

    if t_on.size:
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()


def compare_transition_features(
    df,
    features,
    flag_col="is_long_to_short_isi_onset",
    show_plots=True,
    alpha_normality=0.05,
    alpha_var=0.05,
):
    """
    Compare waveform features between transition and non-transition spikes with
    assumption diagnostics and automatic test selection.

    For each feature:
      • Runs normality tests (Shapiro–Wilk) on each group (subsampled to ≤5000 for stability).
      • Runs Levene’s test for equal variances.
      • Chooses test by rules:
          - If both groups look roughly normal (Shapiro p > alpha_normality OR n > 30)
            and variances are equal (Levene p > alpha_var) → independent t-test.
          - If both groups look roughly normal but variances differ → Welch’s t-test.
          - Otherwise → Mann–Whitney U (nonparametric).
      • Optionally renders three assumption plots per feature:
          (1) histogram + KDE per group, (2) QQ-plot overlay, (3) side-by-side boxplot.
      • Prints a concise decision line per feature (which test and why).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the feature columns and a boolean flag column identifying
        transition spikes (e.g., from `mark_high_burst_onsets`).
    features : list of str
        Numeric feature column names to compare.
    flag_col : str, default='is_long_to_short_isi_onset'
        Boolean column marking transition spikes (True = transition / onset).
    show_plots : bool, default=True
        If True, shows diagnostic plots (distribution, QQ, boxplot) per feature.
    alpha_normality : float, default=0.05
        Significance threshold for Shapiro–Wilk normality test.
    alpha_var : float, default=0.05
        Significance threshold for Levene’s equal-variance test.

    Returns
    -------
    pd.DataFrame
        One row per feature with:
          - n_transition, n_nontransition
          - median_transition, median_nontransition, (implicit difference via medians)
          - shapiro_p_transition, shapiro_p_nontransition
          - levene_p
          - normal_a, normal_b (boolean: Shapiro p > alpha OR n > 30)
          - equal_var (boolean: Levene p > alpha_var)
          - test ({"t-test", "Welch t-test", "Mann–Whitney"})
          - p_value (from the chosen test)
        The table is sorted by ascending p_value.

    Notes
    -----
    • Normality decision is *size-aware*: groups with n > 30 are treated as “roughly normal”
      for t-test purposes even if Shapiro rejects (CLT heuristics).
    • If variances are unequal but both groups are roughly normal, Welch’s t-test is used.
    • If either group is not roughly normal, Mann–Whitney U is used.
    • Features with < 5 valid values in either group are skipped.
    • Plots require seaborn/matplotlib; statistics require scipy.

    Example
    -------
    >>> feats = ["ramp_amp", "peak_width", "peak_sharpness", "inflection_amp", "exp_lambda"]
    >>> results = compare_transition_features(df_marked, feats)
    >>> results.head()
    """   

    results = []

    for feat in features:
        a = pd.to_numeric(df.loc[df[flag_col]==True, feat], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[df[flag_col]==False, feat], errors="coerce").dropna()
        if len(a) < 5 or len(b) < 5:
            print(f"  Skipping {feat}: too few values.")
            continue

        # --- Shapiro–Wilk normality
        shapiro_a = shapiro(a.sample(min(len(a), 5000), random_state=0))[1]
        shapiro_b = shapiro(b.sample(min(len(b), 5000), random_state=0))[1]
        normal_a = shapiro_a > alpha_normality or len(a) > 30
        normal_b = shapiro_b > alpha_normality or len(b) > 30

        # --- Levene for equal variances
        levene_p = levene(a, b)[1]
        equal_var = levene_p > alpha_var

        # --- Choose test intelligently
        if normal_a and normal_b and equal_var:
            test_name = "t-test"
            stat, p = ttest_ind(a, b, equal_var=True)
            decision = "normal or large n → t-test used"
        elif normal_a and normal_b and not equal_var:
            test_name = "Welch t-test"
            stat, p = ttest_ind(a, b, equal_var=False)
            decision = "unequal variance → Welch t-test used"
        else:
            test_name = "Mann–Whitney"
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            decision = "non-normal → Mann–Whitney used"

        print(f"{feat:20s}: {test_name:13s} (p={p:.3})  "
              f"| normal_a={normal_a}, normal_b={normal_b}, equal_var={equal_var} → {decision}")

        # --- Visualization
        if show_plots:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            fig.suptitle(f"{feat} ({test_name})", fontsize=12, weight="bold")

            # Distribution (histogram + KDE)
            sns.histplot(a, ax=axes[0], kde=True, color="#ff7f0e", label="Transition",
                         stat="density", alpha=0.5)
            sns.histplot(b, ax=axes[0], kde=True, color="#1f77b4", label="Non-transition",
                         stat="density", alpha=0.5)
            axes[0].set_title("Distribution")
            axes[0].legend()

            # QQ plot
            probplot(a, dist="norm", plot=axes[1])
            probplot(b, dist="norm", plot=axes[1])
            axes[1].set_title("QQ plot")

            # Boxplot
            sns.boxplot(data=pd.DataFrame({
                "Transition": a,
                "Non-transition": b
            }), ax=axes[2], palette=["#ff7f0e","#1f77b4"])
            axes[2].set_title("Variance")

            plt.tight_layout()
            plt.show()

        # --- store results
        results.append({
            "feature": feat,
            "n_transition": len(a),
            "n_nontransition": len(b),
            "median_transition": np.median(a),
            "median_nontransition": np.median(b),
            "shapiro_p_transition": shapiro_a,
            "shapiro_p_nontransition": shapiro_b,
            "levene_p": levene_p,
            "normal_a": normal_a,
            "normal_b": normal_b,
            "equal_var": equal_var,
            "test": test_name,
            "p_value": p
        })

    return pd.DataFrame(results).sort_values("p_value")




# ------------------------------------------------------------------------------------------- #
# -------------------- Get LFP windows around/pre/post transition spikes--------------------- #
# ------------------------------------------------------------------------------------------- #




def transition_next_isi_windows(
    df: pd.DataFrame,
    time_col: str = "spk_times_ms",              # spike times in ms
    onset_col: str = "is_long_to_short_isi_onset"
) -> pd.DataFrame:
    """
    For each transition spike (onset_col == True), compute the ISI to the very next spike.
    Returns one row per transition with:
      onset_index, onset_time_s, next_spike_time_s, next_isi_s, has_next
    """
    d = df.sort_values(time_col).reset_index(drop=True).copy()
    t_ms = pd.to_numeric(d[time_col], errors="coerce").to_numpy(float)
    is_onset = d[onset_col].astype(bool).to_numpy()

    rows = []
    n = len(d)
    for i in np.where(is_onset)[0]:
        t0_s = t_ms[i] / 1000.0 if np.isfinite(t_ms[i]) else np.nan
        if i + 1 < n and np.isfinite(t_ms[i+1]):
            tn_s = t_ms[i+1] / 1000.0
            next_isi_s = max(0.0, tn_s - t0_s) if np.isfinite(t0_s) else np.nan
            rows.append({
                "onset_index": int(i),
                "onset_time_s": float(t0_s),
                "next_spike_time_s": float(tn_s),
                "next_isi_s": float(next_isi_s),
                "has_next": True,
            })
        else:
            rows.append({
                "onset_index": int(i),
                "onset_time_s": float(t0_s),
                "next_spike_time_s": np.nan,
                "next_isi_s": np.nan,
                "has_next": False,
            })

    return pd.DataFrame(rows)



def make_transition_table(
    df_marked: pd.DataFrame,
    time_col: str = "spk_times_ms",
    onset_col: str = "is_long_to_short_isi_onset",
) -> pd.DataFrame:
    """
    One row per transition with the immediate next spike.
    Columns: onset_index, onset_time_s, next_time_s, has_next
    """
    d = df_marked.sort_values(time_col).reset_index(drop=True)
    t_ms = pd.to_numeric(d[time_col], errors="coerce").to_numpy(float)
    onset_mask = d[onset_col].astype(bool).to_numpy()

    rows = []
    n = len(d)
    for i in np.where(onset_mask)[0]:
        onset_s = t_ms[i] / 1000.0
        if i + 1 < n and np.isfinite(t_ms[i + 1]):
            next_s = t_ms[i + 1] / 1000.0
            has_next = True
        else:
            next_s = np.nan
            has_next = False
        rows.append({
            "onset_index": int(i),
            "onset_time_s": float(onset_s),
            "next_time_s": float(next_s) if has_next else np.nan,
            "has_next": bool(has_next),
        })
    return pd.DataFrame(rows)


def extract_transition_windows_dynamic(
    lfp_uv: np.ndarray,
    fs: float,
    transitions_df: pd.DataFrame,
    pre_s: float,                  # time BEFORE transition (s)
    post_after_next_s: float,      # time AFTER NEXT spike (s)
) -> tuple[pd.DataFrame, list[np.ndarray], list[np.ndarray]]:
    """
    For each transition with a next spike, build a window:
      [onset_time - pre_s,  next_time + post_after_next_s]
    No padding: windows that fall outside the LFP bounds are skipped.

    Returns
    -------
    dyn_df : DataFrame with columns:
        onset_index, onset_time_s, next_time_s, t_start_s, t_end_s, duration_s, kept=True
    windows_uv : list of 1D arrays (µV), one per kept transition
    times_rel_s : list of 1D arrays (seconds), same length as windows; 0 at transition
    """
    lfp = np.asarray(lfp_uv, float)
    n = lfp.size
    fs = float(fs)

    base = transitions_df.copy()
    base = base[base["has_next"]].reset_index(drop=True)
    if base.empty:
        return base.assign(kept=False), [], []

    # accept either column name from upstream code
    next_col = "next_time_s" if "next_time_s" in base.columns else "next_spike_time_s"
    if next_col not in base.columns:
        raise KeyError("transitions_df must contain 'next_time_s' or 'next_spike_time_s'.")

    out_rows, win_list, t_list = [], [], []

    for _, r in base.iterrows():
        t0 = float(r["onset_time_s"])
        tn = float(r[next_col])   # unify downstream as next_time_s

        t_start = t0 - pre_s
        t_end   = tn + post_after_next_s

        i0 = int(round(t_start * fs))
        i1 = int(round(t_end   * fs))

        # keep only fully inside the recording
        if i0 < 0 or i1 > n or i1 <= i0:
            continue

        seg = lfp[i0:i1]
        t_rel = (np.arange(i0, i1) / fs) - t0

        win_list.append(seg)
        t_list.append(t_rel)
        out_rows.append({
            "onset_index": int(r["onset_index"]),
            "onset_time_s": t0,
            "next_time_s": tn,          # standardized name
            "t_start_s": float(t_start),
            "t_end_s": float(t_end),
            "duration_s": float(t_end - t_start),
            "kept": True,
        })

    dyn_df = pd.DataFrame(out_rows)
    return dyn_df, win_list, t_list

def plot_transition_heatmap(
    windows: List[np.ndarray],
    times:   List[np.ndarray],
    title: str = "Transition windows (µV)",
    # pass per-event times (s, relative to transition) to mark on each row
    markers: Optional[Dict[str, np.ndarray]] = None,
    # per-marker style dictionaries
    marker_style: Optional[Dict[str, dict]] = None,
    n_grid: int = 1200,
    cmap: str = "viridis",
):
    """
    Heatmap for variable-length, transition-centered LFP windows (µV).
    Each window has its own timebase (seconds), with 0 at transition.

    Always draws a vertical dashed line at t=0 (transition). You can add a
    'Next spike' marker vector of the same length as the number of events.
    """
    if not isinstance(windows, (list, tuple)) or not isinstance(times, (list, tuple)):
        raise ValueError("Provide windows and times as lists (variable-length mode).")
    if len(windows) == 0 or len(windows) != len(times):
        raise ValueError("windows and times must be non-empty and same length.")

    # sort by duration to make the heatmap easier to read
    durs = np.array([tt[-1] - tt[0] if len(tt) else 0.0 for tt in times], float)
    order = np.argsort(durs)
    win_list = [np.asarray(windows[i], float) for i in order]
    t_list   = [np.asarray(times[i],   float) for i in order]

    # common grid for display only (no padding of data)
    tmin = min(tt[0] for tt in t_list)
    tmax = max(tt[-1] for tt in t_list)
    grid = np.linspace(tmin, tmax, int(n_grid))

    M = np.full((len(win_list), grid.size), np.nan, float)
    for r, (sig, tt) in enumerate(zip(win_list, t_list)):
        if sig.size == 0 or tt.size == 0:
            continue
        m = (grid >= tt[0]) & (grid <= tt[-1])
        M[r, m] = np.interp(grid[m], tt, sig)

    # robust color limits from 5–95%
    finite_vals = M[np.isfinite(M)]
    if finite_vals.size:
        vmin = np.percentile(finite_vals, 5)
        vmax = np.percentile(finite_vals, 95)
        if np.isclose(vmin, vmax):
            pad = 1e-6 if vmax == 0 else 0.05 * abs(vmax)
            vmin, vmax = vmin - pad, vmax + pad
    else:
        vmin, vmax = -1, 1

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    im = ax.imshow(
        M, aspect="auto", origin="lower", cmap=cmap,
        extent=[grid[0], grid[-1], 0, M.shape[0]],
        vmin=vmin, vmax=vmax
    )

    # vertical line at transition (t=0)
    ax.axvline(0, color="w", lw=1.2, ls="--", label="Transition")

    # markers (e.g., next spike)
    default_styles = {
        "Transition": {"s": 28, "facecolors": "white",  "edgecolors": "black", "lw": 0.9, "zorder": 6},
        "Next spike": {"s": 28, "facecolors": "#ff7f0e","edgecolors": "black", "lw": 0.9, "zorder": 6},
    }
    marker_style = {} if marker_style is None else {**default_styles, **marker_style}

    if markers:
        y_rows = np.arange(M.shape[0]) + 0.5
        for name, arr in markers.items():
            arr = np.asarray(arr, float)
            if arr.size != len(windows):
                raise ValueError(f"Marker '{name}' length ({arr.size}) must equal number of events ({len(windows)})")
            # reorder to current (sorted) display order
            arr_sorted = arr[order]
            finite = np.isfinite(arr_sorted) & (arr_sorted >= grid[0]) & (arr_sorted <= grid[-1])
            ax.scatter(arr_sorted[finite], y_rows[finite], label=name, **marker_style.get(name, {}))

    ax.set_ylabel("Events (sorted by duration)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("LFP (µV)")
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))

    # make legend frame slightly dark so white markers are visible
    leg = ax.legend(loc="upper right", fontsize=8,frameon=True)
    if leg:
        leg.get_frame().set_facecolor((0, 0, 0, 0.25))
        leg.get_frame().set_edgecolor("black")

    for text in leg.get_texts():
        text.set_color("white")
    plt.tight_layout()
    plt.show()




def make_random_control_table(
    df_marked: pd.DataFrame,
    time_col: str = "spk_times_ms",
    onset_col: str = "is_long_to_short_isi_onset",
    n_controls: int = 50,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Sample random non-transition spikes that have a valid next spike.
    Returns a DataFrame with the same columns as transition_next_isi_windows:
      onset_index, onset_time_s, next_spike_time_s, next_isi_s, has_next
    """
    d = df_marked.sort_values(time_col).reset_index(drop=True).copy()
    t_ms = pd.to_numeric(d[time_col], errors="coerce").to_numpy(float)

    # candidates: NOT transitions, and have a next spike
    is_onset = d[onset_col].astype(bool).to_numpy()
    has_next = np.r_[np.ones(len(d)-1, dtype=bool), False] & np.isfinite(t_ms)
    cand_idx = np.where((~is_onset) & has_next)[0]

    if cand_idx.size == 0:
        return pd.DataFrame(columns=["onset_index","onset_time_s","next_spike_time_s","next_isi_s","has_next"])

    rng = np.random.default_rng(random_state)
    pick = cand_idx if cand_idx.size <= n_controls else rng.choice(cand_idx, size=n_controls, replace=False)

    rows = []
    for i in np.sort(pick):
        t0 = t_ms[i] / 1000.0
        tn = t_ms[i+1] / 1000.0
        if not np.isfinite(t0) or not np.isfinite(tn): 
            continue
        rows.append({
            "onset_index": int(i),
            "onset_time_s": float(t0),
            "next_spike_time_s": float(tn),
            "next_isi_s": float(max(0.0, tn - t0)),
            "has_next": True
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- #

#LFP specparma time resolved analysis
# ------------------------------------------------------------------------------------------- #
def compute_lfp_windows(
    lfp_signal: np.ndarray,
    fs: float,
    # Multitaper only
    window_length_sec: float = 2.0,
    freq_range: Tuple[float, float] = (1, 90),
    n_freqs: int = 256,
    time_bandwidth: float = 2.0,
    decim_factor: int = 10,
    # Specparam fit controls:
    progress: bool = True,          # show SpectralTimeModel tqdm if True
    n_jobs: int = 1,
    # Specparam modes & peak settings:
    aperiodic_mode: Literal["fixed", "knee"] = "fixed",
    periodic_mode: Literal["gaussian", "skewed_gaussian", "cauchy"] = "gaussian",
    peak_width_limits: Tuple[float, float] = (4.0, 8.0),
    max_n_peaks: int = 4,
    min_peak_height: float = 0.0,
    peak_threshold: float = 2.0,
    verbose: bool = False,
    # NEW: optionally return the spectrogram (linear power)
    return_powers: bool = False,
):
    """
    Build multitaper spectra across time for a 1D LFP trace and fit Specparam across bins.

    Returns:
      (model, freqs, window_times)                  if return_powers=False
      (model, freqs, window_times, powers)          if return_powers=True
        - model: SpectralTimeModel (already fit)
        - freqs: (n_freqs,) linear Hz
        - window_times: list[(start_idx, end_idx)] per time bin (samples)
        - powers: (n_bins, n_freqs) linear power
    """
    lfp = np.asarray(lfp_signal, float)
    win_samps  = int(round(window_length_sec * fs))
    step_samps = max(int(decim_factor), 1)

    # Frequency grid & multitaper TFR → spectrogram
    freqs = np.linspace(freq_range[0], freq_range[1], int(n_freqs))
    n_cycles = freqs * float(window_length_sec)
    tb = max(float(time_bandwidth), 2.0)  # keep ≥ ~3 tapers

    epochs = lfp[np.newaxis, np.newaxis, :]  # (1, 1, n_times)
    tfr = mne.time_frequency.tfr_array_multitaper(
        epochs, sfreq=fs, freqs=freqs, n_cycles=n_cycles,
        time_bandwidth=tb, output="power", decim=decim_factor, verbose=False
    )  # (1,1,n_freqs,n_bins)
    spec = np.squeeze(tfr, axis=(0, 1))      # (n_freqs, n_bins)
    powers = spec.T                           # (n_bins, n_freqs)

    # Map each TFR bin back to a (start,end) sample window of length win_samps
    n_bins = powers.shape[0]
    window_times = [(i * step_samps, i * step_samps + win_samps) for i in range(n_bins)]

    # Fit Specparam across time
    model = SpectralTimeModel(
        aperiodic_mode=aperiodic_mode,
        periodic_mode=periodic_mode,
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        peak_threshold=peak_threshold,
        verbose=verbose,
    )
    sp_progress = "tqdm" if progress else None
    model.fit(
        freqs=freqs,
        spectrogram=powers.T,   # (n_freqs, n_bins)
        freq_range=freq_range,
        n_jobs=n_jobs,
        progress=sp_progress,
    )

    if return_powers:
        return model, freqs, window_times, powers
    else:
        return model, freqs, window_times

# -γ-AUC helper (also returns aperiodic offset & exponent & knee) ---



def _parse_aperiodic_params(ap_params):
    """
    Accepts:
      - array-like [offset, exponent]  (fixed)
      - array-like [offset, knee, exponent] (knee)
      - None  -> (nan, nan, nan)
    Returns: (offset, exponent, knee) with knee=np.nan if fixed.
    """
    if ap_params is None:
        return (np.nan, np.nan, np.nan)
    arr = np.asarray(ap_params, dtype=float).ravel()
    if arr.size >= 3:          # knee mode
        off, knee, exp = arr[:3]
        return (off, exp, knee)
    elif arr.size >= 2:        # fixed mode
        off, exp = arr[:2]
        return (off, exp, np.nan)
    else:
        return (np.nan, np.nan, np.nan)

def gamma_auc_and_params_per_window(
    model,
    band: tuple[float, float] = (30, 90),
    space: str = "linear",          # 'linear' or 'log' per your downstream choice
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      ['win_idx', 'gamma_auc', 'aperiodic_offset', 'aperiodic_exponent', 'aperiodic_knee']

    AUC is computed on (full - aperiodic) within `band` using trapezoid rule,
    where `full` & `aperiodic` are fetched via model.get_model(..., space=<space>).

    Works with aperiodic_mode='fixed' (knee=NaN) and 'knee' (offset, exponent, knee filled).
    """
    freqs = np.asarray(model.freqs)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if sel.sum() == 0:
        raise ValueError("gamma band selection is empty for model.freqs")

    n = int(model.n_time_windows)
    iterator = tqdm(range(n), desc="Computing gamma AUC + aperiodic params") if show_progress else range(n)

    out = []
    for i in iterator:
        m = model.get_model(i)
        if m is None:
            out.append((i, np.nan, np.nan, np.nan, np.nan))
            continue

        # components in requested space

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", message="invalid value encountered in log10")
            warnings.simplefilter("ignore", message="Covariance of the parameters could not be estimated")
    
            full = m.get_model(component="full",      space=space)
            ap   = m.get_model(component="aperiodic", space=space)
    

        if full is None or ap is None:
            out.append((i, np.nan, np.nan, np.nan, np.nan))
            continue

        resid = full - ap

        # robust aperiodic extraction
        ap_params = getattr(m, "aperiodic_params_", None)
        if ap_params is None:
            try:
                ap_params = m.get_params("aperiodic_params")
            except Exception:
                ap_params = None

        off, exp, knee = _parse_aperiodic_params(ap_params)

        auc = np.trapz(resid[sel], freqs[sel])
        out.append((i, float(auc), float(off), float(exp), float(knee)))

    return pd.DataFrame(out, columns=[
        "win_idx", "gamma_auc", "aperiodic_offset", "aperiodic_exponent", "aperiodic_knee"
    ])


def run_specparam_on_transition_windows(
    windows: List[np.ndarray],          # µV
    times_rel: List[np.ndarray],        # s (0 at transition)
    fs: float,
    inner_window_sec: float = 0.500,
    freq_range: Tuple[float, float] = (1, 90),
    n_freqs: int = 256,
    time_bandwidth: float = 2.0,
    decim_factor: int = 9,
    progress: bool = True,              # outer loop progress
    n_jobs: int = 1,

    # Specparam controls
    peak_width_limits: Tuple[float, float] = (4, 8),
    max_n_peaks: int = 2,
    min_peak_height: float = 0.0,
    peak_threshold: float = 2.0,
    aperiodic_mode: str = "fixed",      # or "knee"
    periodic_mode: str = "gaussian",
    verbose: bool = False,

    # AUC config 
    gamma_band: Tuple[float, float] = (30, 55),
    gamma_space: str = "log",           # {"log","linear"}
    compute_gamma_auc: bool = True,

    # Collection toggles
    collect_spectra: bool = False,      # return raw spectra per bin (in-memory)
    inner_specparam_progress: bool = False,  # hard-disable Specparam’s internal tqdm
):
    """
    Apply your existing `compute_lfp_windows` per transition window, extract Specparam
    parameters per inner bin, and (optionally) compute gamma AUC inside this function.

    Returns
    -------
    spec_df : pd.DataFrame
        Tidy table with one row per (epoch, bin):
        epoch_id, bin_id, time_rel_s,
        aperiodic_offset, aperiodic_exponent, aperiodic_knee,
        n_peaks, peak_cf, peak_amp, peak_bw, (gamma_auc if requested)
    spectra : list[np.ndarray] or None
        If collect_spectra=True, list of arrays (n_bins, n_freqs) in linear power.
    models  : list[Specparam.SpectralTimeModel]
        One fitted SpectralTimeModel per epoch (same settings you passed).
    """
    rows = []
    spectra_all = [] if collect_spectra else None
    models_all: List = []

    epoch_iter = tqdm(range(len(windows)), desc="Specparam per epoch") if progress \
                 else range(len(windows))

    for eid in epoch_iter:
        sig = np.asarray(windows[eid], float)
        t_rel = np.asarray(times_rel[eid], float)
        if sig.size == 0 or t_rel.size == 0 or sig.size != t_rel.size:
            continue

        # ---- Compute time-binned spectra (multitaper) with NO internal Specparam bars
        model, freqs, win_times, powers = compute_lfp_windows(
            lfp_signal=sig,
            fs=fs,
            window_length_sec=inner_window_sec,
            freq_range=freq_range,
            n_freqs=n_freqs,
            time_bandwidth=time_bandwidth,
            decim_factor=decim_factor,
            progress=False if not inner_specparam_progress else True,
            n_jobs=n_jobs,
            aperiodic_mode=aperiodic_mode,
            periodic_mode=periodic_mode,
            peak_width_limits=peak_width_limits,
            max_n_peaks=max_n_peaks,
            min_peak_height=min_peak_height,
            peak_threshold=peak_threshold,
            verbose=verbose,
            return_powers=True,
        )
        models_all.append(model)
        if collect_spectra:
            spectra_all.append(powers.copy())  # (n_bins, n_freqs), linear

        # ---- bin-center times (relative to transition)
        bin_centers_rel = []
        for (s0, s1) in win_times:
            c = int(round((s0 + s1 - 1) / 2.0))
            c = int(np.clip(c, 0, sig.size - 1))
            bin_centers_rel.append(float(t_rel[c]))
        bin_centers_rel = np.asarray(bin_centers_rel, float)

        # ---- aperiodic params
        ap = model.get_params('aperiodic_params')
        if ap is None or len(ap) == 0:
            continue
        ap = np.asarray(ap, float)
        if ap.ndim == 1:
            ap = ap[None, :]

        # fixed: [offset, exponent]; knee: [offset, knee, exponent]
        offs = ap[:, 0]
        if aperiodic_mode == "knee" and ap.shape[1] >= 3:
            knees = ap[:, 1]
            exps  = ap[:, 2]
        else:
            knees = np.full(offs.shape, np.nan)
            exps  = ap[:, 1]

        # ---- peaks list (robust parsing)
        peaks_list = model.get_params('peak_params')
        n_bins = len(offs)

        # ---- gamma AUC prep
        if compute_gamma_auc:
            freqs_arr = np.asarray(model.freqs)
            sel = (freqs_arr >= gamma_band[0]) & (freqs_arr <= gamma_band[1])
            if sel.sum() == 0:
                raise ValueError("gamma band selection is empty for model.freqs; adjust gamma_band.")

        for b in range(n_bins):
            pk_cf = pk_amp = pk_bw = np.nan
            n_peaks_here = 0

            if peaks_list is not None and b < len(peaks_list):
                peaks_raw = peaks_list[b]
                if peaks_raw is not None:
                    arr = np.asarray(peaks_raw, dtype=float)
                    if arr.ndim == 2 and arr.shape[0] > 0:
                        n_peaks_here = int(arr.shape[0])
                        take = min(arr.shape[1], 3)
                        first = arr[0, :take]
                        if first.size >= 3:
                            pk_cf, pk_amp, pk_bw = first[:3]
                    elif arr.ndim == 1 and arr.size >= 3:
                        if arr.size % 3 == 0:
                            n_peaks_here = int(arr.size // 3)
                            pk_cf, pk_amp, pk_bw = arr[:3]
                        else:
                            n_peaks_here = 1
                            pk_cf, pk_amp, pk_bw = arr[:3]

            gamma_auc = np.nan
            if compute_gamma_auc:
                m_bin = model.get_model(b)
                if m_bin is not None:
                    full = m_bin.get_model(component="full",      space=gamma_space)
                    ap_c = m_bin.get_model(component="aperiodic", space=gamma_space)
                    if full is not None and ap_c is not None:
                        resid = full - ap_c
                        gamma_auc = float(np.trapz(resid[sel], freqs_arr[sel]))

            rows.append({
                "epoch_id": int(eid),
                "bin_id": int(b),
                "time_rel_s": float(bin_centers_rel[b]) if b < bin_centers_rel.size else np.nan,
                "aperiodic_offset": float(offs[b]),
                "aperiodic_exponent": float(exps[b]),
                "aperiodic_knee": float(knees[b]),
                "n_peaks": int(n_peaks_here),
                "peak_cf": float(pk_cf),
                "peak_amp": float(pk_amp),
                "peak_bw": float(pk_bw),
                **({"gamma_auc": gamma_auc} if compute_gamma_auc else {}),
            })

    spec_df = pd.DataFrame(rows)
    return spec_df, (spectra_all if collect_spectra else None), models_all



#plot heatmap for specparam results in transitions
def plot_specparam_transition_heatmap(
    spec_df: pd.DataFrame,
    param: str = "aperiodic_exponent",
    title: str = "Specparam around transitions",
    # optional per-epoch markers in SECONDS (relative to transition)
    markers: Optional[Dict[str, np.ndarray]] = None,   # e.g., {"Transition": np.zeros(n_epochs)}
    marker_style: Optional[Dict[str, dict]] = None,
    n_grid: int = 1200,
    cmap: str = "viridis",
):
    """
    Make a heatmap like your transition LFP heatmap, but for Specparam outputs.

    Expects a tidy DataFrame with columns at least:
      epoch_id, bin_id, time_rel_s, <param>

    Each epoch can have different time coverage (variable-length). We regrid each
    epoch's (time_rel_s, param) onto a common time axis with NaNs outside coverage.
    """
    need_cols = {"epoch_id", "bin_id", "time_rel_s", param}
    missing = need_cols - set(spec_df.columns)
    if missing:
        raise ValueError(f"spec_df is missing required columns: {sorted(missing)}")

    # build epoch-wise lists like your windows/times
    groups = []
    for eid, df_e in spec_df.groupby("epoch_id"):
        df_e = df_e.sort_values("time_rel_s")
        t = df_e["time_rel_s"].to_numpy(dtype=float)
        y = df_e[param].to_numpy(dtype=float)
        # keep only finite & strictly increasing time to avoid interp issues
        finite = np.isfinite(t) & np.isfinite(y)
        t = t[finite]; y = y[finite]
        if t.size >= 2:
            # enforce monotonic increasing times
            order = np.argsort(t)
            t = t[order]; y = y[order]
        groups.append((int(eid), t, y))

    if not groups:
        raise ValueError("No valid epochs to plot.")

    # sort epochs by duration, like your original function
    durations = []
    for eid, t, _ in groups:
        d = (t[-1] - t[0]) if t.size else 0.0
        durations.append((eid, d))
    order_ids = [eid for eid, _ in sorted(durations, key=lambda x: x[1])]

    # map id -> (t, y)
    epoch_map = {eid: (t, y) for (eid, t, y) in groups}
    # global grid
    tmins = [t[0] for (_, t, _) in groups if t.size]
    tmaxs = [t[-1] for (_, t, _) in groups if t.size]
    if not tmins or not tmaxs:
        raise ValueError("No finite time values found to build the grid.")
    tmin, tmax = float(np.min(tmins)), float(np.max(tmaxs))
    grid = np.linspace(tmin, tmax, int(n_grid))

    # fill matrix (epochs x grid) by interpolation within each epoch's support
    M = np.full((len(order_ids), grid.size), np.nan, dtype=float)
    for r, eid in enumerate(order_ids):
        t, y = epoch_map[eid]
        if t.size == 0:
            continue
        inside = (grid >= t[0]) & (grid <= t[-1])
        if inside.any():
            M[r, inside] = np.interp(grid[inside], t, y)

    # robust color scaling
    finite_vals = M[np.isfinite(M)]
    if finite_vals.size:
        vmin = np.percentile(finite_vals, 5)
        vmax = np.percentile(finite_vals, 95)
        if np.isclose(vmin, vmax):
            pad = 1e-6 if vmax == 0 else 0.05 * abs(vmax)
            vmin, vmax = vmin - pad, vmax + pad
    else:
        vmin, vmax = -1, 1

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    im = ax.imshow(
        M, aspect="auto", origin="lower", cmap=cmap,
        extent=[grid[0], grid[-1], 0, M.shape[0]],
        vmin=vmin, vmax=vmax
    )

    # vertical line at transition (t=0)
    ax.axvline(0, color="w", lw=1.2, ls="--", label="Transition")

    # marker styles (same vibe as your LFP plot)
    default_styles = {
        "Transition": {"s": 28, "facecolors": "white",  "edgecolors": "black", "lw": 0.9, "zorder": 6},
        "Next spike": {"s": 28, "facecolors": "#ff7f0e","edgecolors": "black", "lw": 0.9, "zorder": 6},
    }
    style = {**default_styles, **(marker_style or {})}

    # markers are per-epoch scalar times in seconds; reorder them to match display order
    if markers:
        y_rows = np.arange(M.shape[0]) + 0.5
        # we need a vector aligned to epochs in the order we’re plotting
        # If user passes arrays aligned to unique epoch_id order in spec_df, remap.
        # Build mapping from display row -> original epoch_id index within spec_df
        unique_ids_in_df = np.array(sorted(spec_df["epoch_id"].unique()))
        id_to_pos = {eid: i for i, eid in enumerate(unique_ids_in_df)}
        for name, arr in markers.items():
            arr = np.asarray(arr, float)
            # If arr length equals number of unique epochs, assume aligned to unique_ids_in_df order
            if arr.size == unique_ids_in_df.size:
                # reorder to display order_ids
                arr_sorted = np.array([arr[id_to_pos[eid]] if eid in id_to_pos else np.nan for eid in order_ids], float)
            # Otherwise, assume it's already in display order length
            elif arr.size == len(order_ids):
                arr_sorted = arr
            else:
                raise ValueError(
                    f"Marker '{name}' length ({arr.size}) must equal number of epochs "
                    f"({unique_ids_in_df.size}) or current display rows ({len(order_ids)})."
                )
            finite = np.isfinite(arr_sorted) & (arr_sorted >= grid[0]) & (arr_sorted <= grid[-1])
            ax.scatter(arr_sorted[finite], y_rows[finite], label=name, **style.get(name, {}))

    ax.set_ylabel("Events (sorted by duration)")
    ax.set_xlabel("Time (s, relative to transition)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(param)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))

    # darker legend bg so white markers are visible
    leg = ax.legend(loc="upper right", fontsize=8, frameon=True)
    if leg:
        leg.get_frame().set_facecolor((0, 0, 0, 0.25))
        leg.get_frame().set_edgecolor("black")
        for txt in leg.get_texts():
            txt.set_color("white")

    plt.tight_layout()
    plt.show()
