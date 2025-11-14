"""
skp_feat_cluster_analysis.py
----------------------------
Analysis utilities for identifying and visualizing spike feature clustering patterns
and their temporal dynamics.

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

def compare_feature_groups(
    df,
    features,
    group_col,                          # <-- flexible group column
    groups=None,                         # e.g., ("high","low") or ("cluster1","cluster2")
    group_names=None,                    # plot names, e.g., ("High PW","Low PW")
    show_plots=True,
    alpha_normality=0.05,
    alpha_var=0.05,
):
    """
    Compare numerical features across two groups (flexible group column).

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str
        The feature names to compare.
    group_col : str
        Column in `df` that defines the two groups.
    groups : tuple(str,str) or None
        Which two group labels to compare. If None, detects the first two unique values.
    group_names : tuple(str,str) or None
        Optional pretty names for plots. If None, uses values directly.
    """

    import numpy as np
    import pandas as pd
    from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, probplot
    import seaborn as sns
    import matplotlib.pyplot as plt

    results = []

    # Determine groups automatically if not provided
    vals = df[group_col].dropna().unique()
    if groups is None:
        assert len(vals) >= 2, f"Need ≥2 groups in {group_col}, found: {vals}"
        groups = (vals[0], vals[1])
    if group_names is None:
        group_names = groups

    g1, g2 = groups
    name1, name2 = group_names

    print(f"\nComparing groups: {g1} ({name1}) vs {g2} ({name2}) using column '{group_col}'\n")

    for feat in features:
        a = pd.to_numeric(df.loc[df[group_col] == g1, feat], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[df[group_col] == g2, feat], errors="coerce").dropna()
        if len(a) < 5 or len(b) < 5:
            print(f"  Skipping {feat}: too few values.")
            continue

        # --- Shapiro for normality
        shapiro_a = shapiro(a.sample(min(len(a), 5000), random_state=0))[1]
        shapiro_b = shapiro(b.sample(min(len(b), 5000), random_state=0))[1]
        normal_a = shapiro_a > alpha_normality or len(a) > 30
        normal_b = shapiro_b > alpha_normality or len(b) > 30

        # --- Levene for equal variance
        levene_p = levene(a, b)[1]
        equal_var = levene_p > alpha_var

        # --- Choose statistical test
        if normal_a and normal_b and equal_var:
            test_name = "t-test"
            stat, p = ttest_ind(a, b, equal_var=True)
        elif normal_a and normal_b and not equal_var:
            test_name = "Welch t-test"
            stat, p = ttest_ind(a, b, equal_var=False)
        else:
            test_name = "Mann–Whitney"
            stat, p = mannwhitneyu(a, b, alternative="two-sided")

        print(f"{feat:20s}: {test_name:13s} (p={p:.4f})")

        # --- Visualization
        if show_plots:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            fig.suptitle(f"{feat} ({test_name})", fontsize=12, weight="bold")

            # Histogram
            sns.histplot(a, ax=axes[0], kde=True, color="#ff7f0e", label=name1, stat="density", alpha=0.5)
            sns.histplot(b, ax=axes[0], kde=True, color="#1f77b4", label=name2, stat="density", alpha=0.5)
            axes[0].set_title("Distribution")
            axes[0].legend()

            # QQ plot
            probplot(a, dist="norm", plot=axes[1])
            probplot(b, dist="norm", plot=axes[1])
            axes[1].set_title("QQ plot")

            # Boxplot
            sns.boxplot(data=pd.DataFrame({name1: a, name2: b}), ax=axes[2], palette=["#ff7f0e","#1f77b4"])
            axes[2].set_title("Variance")

            plt.tight_layout()
            plt.show()

        results.append({
            "feature": feat,
            "group1": g1, "group2": g2,
            "median_group1": np.median(a),
            "median_group2": np.median(b),
            "shapiro_p_g1": shapiro_a,
            "shapiro_p_g2": shapiro_b,
            "levene_p": levene_p,
            "test": test_name,
            "p_value": p
        })

    return pd.DataFrame(results).sort_values("p_value")


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

   
    plt.tight_layout()
    plt.show()


def cluster_transition_matrix(df: pd.DataFrame, cluster_col: str = "log_isi_cluster") -> pd.DataFrame:
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



# ------------------------------------------------------------------------------------------- #
# ------------------------------ ISI transition functions --------------------- #
# ------------------------------------------------------------------------------------------- #


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



# ------------------------------------------------------------------------------------------- #
# --------------------  Extract LFP windows --------------------- #
# ------------------------------------------------------------------------------------------- #

def extract_lfp_windows(
    spk_df: pd.DataFrame,
    lfp_times_ms: np.ndarray,
    lfp_signal: np.ndarray,
    pre_s: float = 1.0,
    post_s: float = 1.5,
    condition: Optional[str] = None
):
    """
    Extract LFP windows around spikes, optionally filtered by a condition.

    Returns dict with:
      windows: list of arrays (lfp µV)
      times_rel_ms: list of arrays (ms, spike = 0)
      spike_times_ms: list of spike times (ms)
      next_spike_times_ms: list of next spike times (ms)
      meta_df: Dataframe describing each window
    """

    df = spk_df.copy()

    # Apply condition if provided
    if condition is not None:
        df = df.query(condition)

    if df.empty:
        print("No spikes matched the condition — returning empty output.")
        return {
            "windows": [],
            "times_rel_ms": [],
            "spike_times_ms": [],
            "next_spike_times_ms": [],
            "meta_df": pd.DataFrame()
        }

    # Convert pre/post from s → ms
    pre_ms = pre_s * 1000.0
    post_ms = post_s * 1000.0

    # Build helper for spike→next spike lookup
    df_sorted = spk_df.sort_values("spk_id").set_index("spk_id")

    windows = []
    times_rel_ms = []
    spike_times = []
    next_spike_times = []
    meta_rows = []

    for _, row in df.iterrows():

        sid = int(row["spk_id"])
        t0 = float(row["spk_times_ms"])

        # Lookup next spike
        if sid + 1 not in df_sorted.index:
            continue  # skip last spike (no next spike)
        t_next = float(df_sorted.loc[sid + 1, "spk_times_ms"])

        # Window boundaries
        t_start = t0 - pre_ms
        t_end = t0 + post_ms

        # Clip windows to LFP range
        if t_start < lfp_times_ms[0] or t_end > lfp_times_ms[-1]:
            continue

        # Index into LFP vector
        idx0 = np.searchsorted(lfp_times_ms, t_start)
        idx1 = np.searchsorted(lfp_times_ms, t_end)

        seg = lfp_signal[idx0:idx1]
        t_seg = lfp_times_ms[idx0:idx1]

        # build relative time (spike = 0)
        t_rel = t_seg - t0

        windows.append(seg)
        times_rel_ms.append(t_rel)
        spike_times.append(t0)
        next_spike_times.append(t_next)

        meta_rows.append({
            "spk_id": sid,
            "spike_time_ms": t0,
            "next_spike_time_ms": t_next,
            "t_start_ms": t_start,
            "t_end_ms": t_end
        })

    return {
        "windows": windows,
        "times_rel_ms": times_rel_ms,
        "spike_times_ms": spike_times,
        "next_spike_times_ms": next_spike_times,
        "meta_df": pd.DataFrame(meta_rows)
    }





# ------------------------------------------------------------------------------------------- #
# --------------------  LFP time resolved analysis --------------------- #
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


def run_time_resolved_specparam_on_window(
    lfp_window,
    times_rel,
    fs,
    inner_window_sec=0.5,
    freq_range=(1, 90),
    n_freqs=256,
    time_bandwidth=2.0,
    decim_factor=10,
    band_dict=None,
    aperiodic_mode="fixed",
    periodic_mode="gaussian",
    peak_width_limits=(4.0, 8.0),
    max_n_peaks=4,
    min_peak_height=0.0,
    peak_threshold=2.0,
    verbose=False,
    next_spike_rel=None,
):
    """
    Time-resolved Specparam on a single spike-centered LFP window.

    Parameters
    ----------
    lfp_window : 1D array
        LFP samples for one spike-centered window.
    times_rel : 1D array
        Time axis for lfp_window, typically spike-centered.
        Can be in ms or s; this function auto-detects and converts to seconds.
    fs : float
        Sampling rate of the LFP (Hz).
    next_spike_rel : float, optional
        Relative time to the next spike for this window (from your groups dict).
        Returned as-is under key 'next_spike_rel'.

    Returns
    -------
    out : dict
        {
            "t_bins_s": epoch_times_s,      # (n_epochs,)
            "epoch_idx": epoch_idx,         # (n_epochs,)
            "freqs": freqs,                 # (n_freqs,)
            "powers": powers,               # (n_epochs, n_freqs), linear space
            "offset": offset,               # (n_epochs,)
            "exponent": exponent,           # (n_epochs,)
            "knee": knee or None,           # (n_epochs,) or None
            "r_squared": r_squared,         # (n_epochs,)
            "band_aucs": band_aucs,         # dict[band] -> (n_epochs,)
            "next_spike_rel": next_spike_rel,
            "model": time_model,            # SpectralTimeModel
        }
    """

    if band_dict is None:
        band_dict = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "gamma": (30, 80),
        }

    # ---------------------------------------------------------
    # 1) Run multitaper + Specparam time model
    # ---------------------------------------------------------
    time_model, freqs, window_times, powers = compute_lfp_windows(
        lfp_signal=np.asarray(lfp_window, float),
        fs=fs,
        window_length_sec=inner_window_sec,
        freq_range=freq_range,
        n_freqs=n_freqs,
        time_bandwidth=time_bandwidth,
        decim_factor=decim_factor,
        progress=False,               # no Specparam tqdm
        aperiodic_mode=aperiodic_mode,
        periodic_mode=periodic_mode,
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        peak_threshold=peak_threshold,
        verbose=verbose,
        return_powers=True,
    )

    powers = np.asarray(powers, float)          # (n_epochs, n_freqs)
    n_epochs, n_freqs_actual = powers.shape
    freqs = np.asarray(freqs, float)

    # ---------------------------------------------------------
    # 2) Build epoch time axis in *seconds*, using your times_rel
    #    - times_rel is sample-wise time for the raw window (ms or s)
    #    - window_times are (start_idx, end_idx) in samples
    # ---------------------------------------------------------
    times_rel = np.asarray(times_rel, float)

    # crude but robust: if range is big, assume ms and convert
    if np.nanmax(np.abs(times_rel)) > 20.0:
        times_rel_s = times_rel / 1000.0
    else:
        times_rel_s = times_rel

    sample_idx = np.arange(times_rel_s.size)
    epoch_centers = np.array([0.5 * (s + e) for (s, e) in window_times])
    epoch_times_s = np.interp(epoch_centers, sample_idx, times_rel_s)

    epoch_idx = np.arange(n_epochs, dtype=int)

    # ---------------------------------------------------------
    # 3) Extract aperiodic params & r_squared from the time model
    # ---------------------------------------------------------
    aperiodic_params = np.asarray(time_model.get_params("aperiodic_params"))

    if aperiodic_params.shape[1] == 2:
        offset = aperiodic_params[:, 0]
        exponent = aperiodic_params[:, 1]
        knee = None
    else:
        offset = aperiodic_params[:, 0]
        knee = aperiodic_params[:, 1]
        exponent = aperiodic_params[:, 2]

    try:
        r_squared = np.asarray(time_model.get_params("r_squared")).ravel()
    except Exception:
        r_squared = np.full(n_epochs, np.nan, dtype=float)

    # ---------------------------------------------------------
    # 4) Band AUCs: area between full & aperiodic (log10 space)
    #    For each epoch:
    #       - get child SpectralModel via get_model(ind=epoch_i)
    #       - get_data('full', 'log') & get_data('aperiodic', 'log')
    #       - AUC = ∫(full_log - ape_log) df over the band
    # ---------------------------------------------------------
    band_aucs = {bname: np.full(n_epochs, np.nan, dtype=float)
                 for bname in band_dict.keys()}
   

    for ei in range(n_epochs):
        
        try:
            full_log = time_model.get_model(ind = ei).get_model(component="full",      space="log")
            ape_log  = time_model.get_model(ind = ei).get_model(component="aperiodic", space="log")
            
           
        except Exception:
            continue

        if full_log is None or ape_log is None:
            continue

        full_log = np.asarray(full_log, float)
        ape_log  = np.asarray(ape_log, float)
        

        # Specparam SpectralModel get_data should return 1D arrays
        if full_log.shape != freqs.shape or ape_log.shape != freqs.shape:
            # Something is inconsistent; skip this epoch
            continue

        for bname, (f_lo, f_hi) in band_dict.items():
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            if not np.any(mask):
                continue

            diff = full_log[mask] - ape_log[mask]
            auc = np.trapz(diff, freqs[mask])
            band_aucs[bname][ei] = float(auc)

    # ---------------------------------------------------------
    # 5) Pack and return
    # ---------------------------------------------------------
    return {
        "t_bins_s": epoch_times_s,          # (n_epochs,)
        "epoch_idx": epoch_idx,            # (n_epochs,)
        "freqs": freqs,                    # (n_freqs,)
        "powers": powers,                  # (n_epochs, n_freqs), linear
        "offset": offset,                  # (n_epochs,)
        "exponent": exponent,              # (n_epochs,)
        "knee": knee,                      # (n_epochs,) or None
        "r_squared": r_squared,            # (n_epochs,)
        "band_aucs": band_aucs,            # dict[band] -> (n_epochs,)
        "next_spike_rel": next_spike_rel,  # scalar from groups (ms or s, your choice)
        "model": time_model,               # SpectralTimeModel
    }


def run_time_resolved_specparam_for_groups(
    groups,
    fs,
    inner_window_sec=0.5,
    freq_range=(1, 90),
    n_freqs=256,
    time_bandwidth=2.0,
    decim_factor=10,
    band_dict=None,
    aperiodic_mode="fixed",
    periodic_mode="gaussian",
    peak_width_limits=(4.0, 8.0),
    max_n_peaks=4,
    min_peak_height=0.0,
    peak_threshold=2.0,
    verbose=False,
    max_windows_per_group=None,
):
    results = {}

    for group_name in tqdm(groups.keys(), desc="Groups"):
        group = groups[group_name]

        windows        = list(group["windows"])
        times_rel_list = list(group["times_rel"])
        next_rel_list  = list(group["next_rel"])   # for heatmaps / summaries

        if max_windows_per_group is not None:
            windows        = windows[:max_windows_per_group]
            times_rel_list = times_rel_list[:max_windows_per_group]
            next_rel_list  = next_rel_list[:max_windows_per_group]

        if len(windows) == 0:
            results[group_name] = []
            continue

        group_out = []

        for w_i, win in enumerate(tqdm(windows, desc=f"{group_name} windows", leave=False)):

            out_w = run_time_resolved_specparam_on_window(
                lfp_window=win,
                times_rel=times_rel_list[w_i],
                fs=fs,
                inner_window_sec=inner_window_sec,
                freq_range=freq_range,
                n_freqs=n_freqs,
                time_bandwidth=time_bandwidth,
                decim_factor=decim_factor,
                band_dict=band_dict,
                aperiodic_mode=aperiodic_mode,
                periodic_mode=periodic_mode,
                peak_width_limits=peak_width_limits,
                max_n_peaks=max_n_peaks,
                min_peak_height=min_peak_height,
                peak_threshold=peak_threshold,
                verbose=verbose,
                next_spike_rel=next_rel_list[w_i],   # stored inside out_w
            )

            group_out.append(out_w)

        results[group_name] = group_out

    return results

# ------------------------------------------------------------------------------------------- #
# --------------------  Time-resolved and window visualziations  --------------------- #
# ------------------------------------------------------------------------------------------- #

def plot_window_feature_groups_heatmap(
    groups: Dict[str, Dict[str, np.ndarray]],
    *,
    feature_label: str = "value",
    cmap: str = "viridis",
    n_grid: int = 800,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    time_unit: str = "s",
    xlim: Optional[Tuple[float, float]] = None,
):
    """
    Plot heatmaps for one or more groups of spike-centred windows, for ANY
    time-series feature aligned to those windows.

    Parameters
    ----------
    groups : dict
        {
          "Group name": {
              "windows":   list of 1D arrays (feature traces),
              "times_rel": list of 1D arrays (same length as windows, time rel. to spike),
              "next_rel":  1D array (len = n_windows) with next-spike time (same units),
          },
          ...
        }
    feature_label : str
        Label for colorbar (e.g. "LFP (µV)", "gamma AUC").
    cmap : str
        Matplotlib colormap name.
    n_grid : int
        Number of time bins in common grid for display.
    vmin, vmax : float or None
        Color scale limits. If None, computed from all groups together (5–95%).
    time_unit : {"s","ms"}
        Units of the times in `times_rel` and `next_rel` (for axis label only).
    xlim : (float, float) or None
        Optional x-axis limits (in the same units as times_rel). If None, use
        the full grid across all groups.

    Returns
    -------
    fig : Figure
    axes : list[Axes]
    out  : dict
        {"Group name": {"grid_t": 1D grid, "matrix": 2D (events × time)}}
    """
    # ---------- collect all times to build global grid ----------
    all_tmins, all_tmaxs = [], []
    for gname, gdict in groups.items():
        times_list = gdict["times_rel"]
        for t in times_list:
            t = np.asarray(t, float)
            if t.size == 0 or not np.any(np.isfinite(t)):
                continue
            all_tmins.append(np.nanmin(t))
            all_tmaxs.append(np.nanmax(t))

    if not all_tmins or not all_tmaxs:
        raise ValueError("No finite times found in any group.")

    g_tmin = float(np.min(all_tmins))
    g_tmax = float(np.max(all_tmaxs))
    grid_t = np.linspace(g_tmin, g_tmax, int(n_grid))

    # ---------- first pass: build matrices & gather global feature range ----------
    matrices = {}
    all_vals = []

    for gname, gdict in groups.items():
        windows   = gdict["windows"]
        times_rel = gdict["times_rel"]

        n_ev = len(windows)
        M = np.full((n_ev, grid_t.size), np.nan, float)

        for i, (w, t) in enumerate(zip(windows, times_rel)):
            w = np.asarray(w, float)
            t = np.asarray(t, float)
            if w.size == 0 or t.size == 0 or w.size != t.size:
                continue
            inside = (grid_t >= t[0]) & (grid_t <= t[-1])
            if not inside.any():
                continue
            M[i, inside] = np.interp(grid_t[inside], t, w)

        matrices[gname] = M
        all_vals.append(M[np.isfinite(M)])

    # global color limits if not provided
    if all_vals:
        all_vals = np.concatenate(all_vals)
        if vmin is None or vmax is None:
            vmin_q = np.percentile(all_vals, 5)
            vmax_q = np.percentile(all_vals, 95)
            if vmin is None:
                vmin = vmin_q
            if vmax is None:
                vmax = vmax_q

    # ---------- plotting ----------
    n_groups = len(groups)
    fig, axes = plt.subplots(
        n_groups, 1,
        figsize=(9.0, 2.8 * n_groups),
        sharex=True,
        constrained_layout=True,   # nicer than tight_layout with colorbar
    )

    if n_groups == 1:
        axes = [axes]

    out = {}
    group_names = list(groups.keys())

    # main heatmaps
    im = None
    for ax, gname in zip(axes, group_names):
        M = matrices[gname]
        out[gname] = {"grid_t": grid_t, "matrix": M}

        im = ax.imshow(
            M,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[grid_t[0], grid_t[-1], 0, M.shape[0]],
            vmin=vmin,
            vmax=vmax,
        )

        # vertical dashed line at the spike (t=0)
        ax.axvline(0, color="w", ls="--", lw=1.5, label="Spike")

        # next-spike markers (if provided)
        next_rel = groups[gname].get("next_rel", None)
        if next_rel is not None:
            next_rel = np.asarray(next_rel, float)
            # we assume 1 value per event
            y = np.arange(M.shape[0]) + 0.5
            valid = np.isfinite(next_rel) & (next_rel >= grid_t[0]) & (next_rel <= grid_t[-1])
            ax.scatter(
                next_rel[valid],
                y[valid],
                s=16,
                facecolors="none",
                edgecolors="pink",
                linewidths=1.0,
                label="Next spike",
            )

        ax.set_ylabel("Events")
        ax.set_title(gname)

        # small legend (only spike & next spike)
        leg = ax.legend(loc="upper left", fontsize=8, frameon=True)
        if leg is not None:
            leg.get_frame().set_facecolor((0, 0, 0, 0.25))
            leg.get_frame().set_edgecolor("black")
            for txt in leg.get_texts():
                txt.set_color("white")

    # x-axis label on the last subplot
    unit_str = "s" if time_unit == "s" else "ms"
    axes[-1].set_xlabel(f"Time ({unit_str}, relative to spike)")

    # optional x-limits (useful to cut off empty white space)
    if xlim is not None:
        for ax in axes:
            ax.set_xlim(*xlim)

    # one shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label(feature_label)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))

    return fig, axes, out




def plot_window_feature_group_traces(
    groups: Dict[str, Dict[str, Any]],
    *,
    time_unit: str = "ms",          # "ms" or "s" for the values in times_rel & next_rel
    ylabel: str = "value",
    title: str = "Average feature around spikes",
    band_k: float = 1.0,            # how many SDs for the shaded band
    band_alpha: float = 0.22,
    colors: Optional[Dict[str, str]] = None,   # optional: {"Group name": "#hex"}
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Dict[str, np.ndarray]]]:
    """
    Plot average ± SD traces for one or more groups of spike–centered windows.

    Parameters
    ----------
    groups : dict
        {"Group name": {"windows": [...], "times_rel": [...], "next_rel": [...]} }
        - windows   : list of 1D arrays (feature values per window)
        - times_rel : list of 1D arrays (same length as each window)
        - next_rel  : 1D array/list of scalar next-spike times (same length as windows)
    time_unit : {"ms","s"}
        Unit of times_rel and next_rel.
    band_k : float
        Multiplier for SD (1.0 = 1×SD band).
    """

 

    if time_unit not in ("ms", "s"):
        raise ValueError("time_unit must be 'ms' or 's'.")

    # ---------- build a common time grid from the first non-empty group ----------
    base_T = None
    for g in groups.values():
        t_list = g.get("times_rel", [])
        if t_list and len(t_list[0]) > 1:
            base_T = np.asarray(t_list[0], float)
            break

    if base_T is None:
        raise ValueError("No non-empty times_rel found in any group.")

    # convert to seconds if needed
    if time_unit == "ms":
        Tgrid = base_T / 1000.0
    else:
        Tgrid = base_T.copy()

    # ---------- set up plotting ----------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    out: Dict[str, Dict[str, np.ndarray]] = {}

    # color cycle
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    def _get_color(i, name):
        if colors and name in colors:
            return colors[name]
        if default_colors:
            return default_colors[i % len(default_colors)]
        return None  # let matplotlib choose

    # ---------- process each group ----------
    for gi, (name, info) in enumerate(groups.items()):
        windows   = info.get("windows", [])
        times_rel = info.get("times_rel", [])
        next_rel  = np.asarray(info.get("next_rel", []), float)

        if len(windows) == 0 or len(times_rel) == 0:
            print(f"Group '{name}': empty; skipping.")
            continue

        if len(windows) != len(times_rel):
            raise ValueError(
                f"Group '{name}': len(windows) ({len(windows)}) "
                f"!= len(times_rel) ({len(times_rel)})."
            )

        # Interpolate each window onto Tgrid
        mats = []
        for w, t in zip(windows, times_rel):
            w = np.asarray(w, float)
            t = np.asarray(t, float)
            if w.size != t.size or w.size < 2:
                continue

            # convert this window's timebase to seconds
            if time_unit == "ms":
                t_sec = t / 1000.0
            else:
                t_sec = t

            # only interpolate where this window actually has support
            yi = np.full_like(Tgrid, np.nan, dtype=float)

            # find overlapping region between the Tgrid and this window
            left  = max(Tgrid[0], t_sec[0])
            right = min(Tgrid[-1], t_sec[-1])
            
            if right > left:
                mask = (Tgrid >= left) & (Tgrid <= right)
                yi[mask] = np.interp(Tgrid[mask], t_sec, w)
                mats.append(yi)

        if len(mats) == 0:
            print(f"Group '{name}': no windows overlapped the common time grid; skipping.")
            continue

        A = np.stack(mats, axis=0)          # (n_windows, n_time)
        mean = np.nanmean(A, axis=0)
        sd   = np.nanstd(A, axis=0)

        col = _get_color(gi, name)
        ax.plot(Tgrid, mean, lw=2.0, color=col, label=name)
        ax.fill_between(
            Tgrid, mean - band_k * sd, mean + band_k * sd,
            color=col, alpha=band_alpha, linewidth=0
        )

        # mean next-spike time for this group
        mean_next_s = np.nan
        if next_rel.size:
            if time_unit == "ms":
                next_s = next_rel / 1000.0
            else:
                next_s = next_rel
            mean_next_s = float(np.nanmean(next_s))
            ax.axvline(mean_next_s, color=col, ls=":", lw=1.7)

        out[name] = {
            "time_s": Tgrid,
            "mean": mean,
            "std": sd,
            "mean_next_rel_s": mean_next_s,
            "n_windows": A.shape[0],
        }

    # spike at 0
    ax.axvline(0.0, color="k", ls="--", lw=2.0, label="spike")

    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()

    return fig, ax, out


def plot_aperiodic_fit_with_band_auc(
    freqs,
    powers,
    time_model,
    epoch_i: int = 0,
    band_range=(30.0, 80.0),
    title: str = None,
    log_freq: bool = False,
    ax=None,
):
    """
    Plot raw PSD, Specparam full fit, aperiodic fit, and shade the band AUC
    (area between full and aperiodic in log10 power) for a single epoch.

    Parameters
    ----------
    freqs : 1D array
        Frequency grid (Hz), in linear space. Shape (n_freqs,).
    powers : 2D array
        Linear power spectrogram, shape (n_epochs, n_freqs).
    time_model : SpectralTimeModel
        The fitted SpectralTimeModel from compute_lfp_windows.
    epoch_i : int
        Index of the time bin / epoch to plot.
    band_range : (float, float)
        Frequency band over which to compute & shade AUC (Hz).
    title : str or None
        Figure title.
    log_freq : bool
        If True, plot x-axis in log scale.
    ax : matplotlib Axes or None
        Optional axis to draw on. If None, creates a new figure & axis.

    Returns
    -------
    fig, ax, auc
        fig : Figure
        ax  : Axes
        auc : float, band AUC (log10 space) between full & aperiodic in band_range
    """

    freqs = np.asarray(freqs, float)
    P_lin = np.asarray(powers[epoch_i], float)

    # ---------- grab the child SpectralModel for this epoch ----------
    # This gives you a 'group' object containing just this one time point
    child_time = time_model.get_group([epoch_i], output_type="group")
    child_model = child_time.get_model(ind=0, regenerate=False)  # SpectralModel

    # full & aperiodic components in log10 power
    full_log = time_model.get_model(epoch_i).get_model(component="full",      space="log")
    ape_log  = time_model.get_model(epoch_i).get_model(component="aperiodic", space="log")

    if full_log is None or ape_log is None:
        raise RuntimeError(
            f"No full / aperiodic data available for epoch {epoch_i}. "
            "Check that the model fit completed successfully."
        )

    full_log = np.asarray(full_log, float)
    ape_log  = np.asarray(ape_log,  float)

    if full_log.shape != freqs.shape or ape_log.shape != freqs.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"  freqs:    {freqs.shape}\n"
            f"  full_log: {full_log.shape}\n"
            f"  ape_log:  {ape_log.shape}"
        )

    # ---------- compute band AUC in log space ----------
    f_lo, f_hi = band_range
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)

    if not np.any(band_mask):
        raise ValueError(f"No frequencies within band {band_range} Hz")

    diff = full_log[band_mask] - ape_log[band_mask]
    auc = np.trapz(diff, freqs[band_mask])   # log10-power area

    # ---------- plotting ----------
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # raw multitaper (just to check it's consistent)
    raw_log = np.log10(P_lin)
    ax.plot(freqs, raw_log, lw=1.5, alpha=0.4, label="Raw multitaper")

    # specparam full & aperiodic
    ax.plot(freqs, full_log, lw=2.0, label="Specparam full")
    ax.plot(freqs, ape_log,  lw=2.0, ls="--", label="Aperiodic fit")

    # shaded band area
    ax.fill_between(
        freqs[band_mask],
        full_log[band_mask],
        ape_log[band_mask],
        alpha=0.3,
        label=f"{f_lo:.0f}–{f_hi:.0f} Hz AUC\n= {auc:.3f}",
    )

    if log_freq:
        ax.set_xscale("log")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (log10)")
    if title is None:
        title = f"Epoch {epoch_i} | AUC {f_lo:.0f}–{f_hi:.0f} Hz = {auc:.3f}"
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()

    return fig, ax, auc
# ------------------------------------------------------------------------------------------- #
# --------------------  Time-resolved post specparam analysis  --------------------- #
# ------------------------------------------------------------------------------------------- #




def make_feature_groups(time_res_results, feature, band=None):
    """
    Convert time_res_results (from run_time_resolved_specparam_for_groups)
    into a groups-style dict for heatmap / trace plotting.

    Parameters
    ----------
    time_res_results : dict
        {
            group_name: [
                {
                    "t_bins_s": ...,
                    "offset": ...,
                    "exponent": ...,
                    "r_squared": ...,
                    "knee": ...,
                    "band_aucs": {...},
                    "next_spike_rel": float,
                    "epoch_idx": np.ndarray,
                    ...
                },
                ...
            ],
            ...
        }

    feature : str
        One of:
            "offset"
            "exponent"
            "r_squared"
            "knee"
            "band"   (requires band="gamma"/"alpha"/etc.)

    band : str or None
        Only used if feature == "band".

    Returns
    -------
    feat_groups : dict
        {
            group_name: {
                "windows":    [ feature-trace for each window (1D array) ],
                "times_rel":  [ corresponding t_bins_s arrays (1D) ],
                "next_rel":   [ next_spike_rel scalar per window ] (np.array)
                "epoch_idx":  [ epoch_idx per window ] (optional, if present)
            }
        }
    """

    feat_groups = {}

    for group_name, win_list in time_res_results.items():
        if win_list is None or len(win_list) == 0:
            continue

        windows_feat  = []
        windows_times = []
        windows_next  = []
        windows_epoch_idx = []

        for w in win_list:

            t_bins   = np.asarray(w["t_bins_s"])
            next_rel = w["next_spike_rel"]/1000

            # ---------- PICK FEATURE ----------
            if feature in ["offset", "exponent", "r_squared", "knee"]:
                arr = w.get(feature, None)
                if arr is None:
                    continue

            elif feature == "band":
                if band is None:
                    raise ValueError("If feature=='band', you must pass band='gamma'/'theta'/etc.")
                arr = w["band_aucs"].get(band, None)
                if arr is None:
                    continue

            else:
                raise ValueError(f"Unknown feature '{feature}'.")

            arr = np.asarray(arr)

            # safety: ensure array and time lengths match
            if arr.shape[0] != t_bins.shape[0]:
                # skip weird cases
                continue

            windows_feat.append(arr)
            windows_times.append(t_bins)
            windows_next.append(next_rel)

            # optional: keep epoch indices if present
            if "epoch_idx" in w:
                windows_epoch_idx.append(np.asarray(w["epoch_idx"]))
            else:
                windows_epoch_idx.append(None)

        if len(windows_feat) == 0:
            # no valid windows for this group
            continue

        feat_groups[group_name] = {
            "windows":   windows_feat,
            "times_rel": windows_times,
            "next_rel":  np.asarray(windows_next, float),
            "epoch_idx": windows_epoch_idx,     # you can ignore this if you don't care
        }

    return feat_groups