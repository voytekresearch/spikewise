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
from typing import Optional, List, Tuple, Dict, Any
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
            print(f"⚠️  Skipping {feat}: too few values.")
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



