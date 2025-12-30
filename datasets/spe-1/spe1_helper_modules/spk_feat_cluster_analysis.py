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
from tqdm.auto import tqdm
import mne
from specparam import SpectralTimeModel
from typing import Optional, List, Tuple, Dict, Any, Literal, Callable, Union, Sequence
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, probplot, f_oneway




# ------------------------------------------------------------------------------------------- #
# ------------------------------ Global cluster color registry ------------------------------ #
# ------------------------------------------------------------------------------------------- #

CLUSTER_COLORS = {
    "low":  "#1f77b4",   # blue
    "mid":  "#2ca02c",   # green
    "high": "#ff7f0e",   # orange
}

def get_cluster_color(label: str, fallback: str = "black") -> str:
    """Return consistent color for a cluster label."""
    return CLUSTER_COLORS.get(str(label), fallback)

# ------------------------------------------------------------------------------------------- #
# ------------------------------ Cluster features that show grouped data --------------------- #
# ------------------------------------------------------------------------------------------- #
# Post-process: remove first/last 0.5s epochs
def trim_edges(results, edge_sec=0.5):
    trimmed = []
    
    for out_w in results:
        t_bins = out_w["t_bins_s"]
        mask = (t_bins >= (t_bins.min() + edge_sec)) & (t_bins <= (t_bins.max() - edge_sec))
        
        # Apply mask to all epoch-wise arrays
        out_w_trimmed = {
            **out_w,
            "t_bins_s": out_w["t_bins_s"][mask],
            "epoch_idx": out_w["epoch_idx"][mask],
            "powers": out_w["powers"][mask, :],
            "offset": out_w["offset"][mask],
            "exponent": out_w["exponent"][mask],
            "r_squared": out_w["r_squared"][mask],
            "band_aucs": {b: vals[mask] for b, vals in out_w["band_aucs"].items()},
        }
        # knee may be None or array
        if out_w["knee"] is not None:
            out_w_trimmed["knee"] = out_w["knee"][mask]
        trimmed.append(out_w_trimmed)
    return trimmed


def plot_spike_clusters_from_df(
    df,
    sp,
    cluster_col,
    mode="full",
    plot_average=True,
    plot_average_std=True,
    colors=None,
    title=None,
    figsize=(14, 4),
):
    """
    Plot spike waveforms grouped by clusters using df['spk_id'] to index spikes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
            cluster_col : cluster labels (e.g. 'peak_width_cluster')
            'spk_id'    : the true spike index into sp.spikes
    sp : Spike
        The Spike object containing the actual waveforms
    cluster_col : str
        Cluster feature column name
    """

    # Extract cluster labels
    labels = sorted(df[cluster_col].dropna().unique().tolist())


    # Default color scheme
    colors = {lab: get_cluster_color(lab) for lab in labels}


    # Build index groups based on df['spk_id']
    ind_groups = []
    group_names = []

    for lab in labels:
        inds = df.loc[df[cluster_col] == lab, "spk_id"].astype(int).tolist()
        if len(inds) > 0:
            ind_groups.append(inds)
            group_names.append(str(lab))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Use the Spike class’ existing plotting engine
    sp.plot(
        inds=None,                   # we supply groups below
        mode=mode,
        in_ms=True,
        show_points=False,
        ax=ax,
        groups=True,                 # activates group overlays
        ind_groups=ind_groups,       # list of lists of spike indices
        group_names=group_names,
        plot_average=plot_average,
        plot_average_std=plot_average_std,
    )

    # Update average line colors based on cluster colors
    for line in ax.lines:
        lab = line.get_label()
        for cluster_label in labels:
            if cluster_label in lab:
                line.set_color(colors[cluster_label])

    ax.set_title(title or f"Spike waveforms grouped by {cluster_col}")
    plt.tight_layout()

    return fig, ax


def plot_full_cluster_report(
    df: pd.DataFrame,
    sp,
    cluster_col: str,
    *,
    time_col: str = "spk_times_ms",
    time_unit: str = "ms",
    bin_size_ms: int = 1000,
    sigma_bins: int = 2,
    heatmap_cmap: str = "magma",
):
    """
    Full diagnostic plotting report for a given cluster column.
    """

    SPIKE_WAVEFORM_FEATURES = [
        "ramp_amp",
        "inflection_time",
        "inflection_amp",
        "peak_amp",
        "peak_width",
        "peak_sharpness",
        "exp_lambda",
        "exp_const",
        "log_isi",
    ]

    if cluster_col not in df.columns:
        raise KeyError(f"{cluster_col} not found in dataframe")

    print(f"\n===== CLUSTER REPORT: {cluster_col} =====")

    # --------------------------------------------------
    # A) Spike waveforms
    # --------------------------------------------------
    print("→ Plotting spike waveforms by cluster")
    plot_spike_clusters_from_df(df, sp, cluster_col)

    # --------------------------------------------------
    # B) Cluster proportions over time
    # --------------------------------------------------
    print("→ Plotting cluster proportions over time")
    plot_clusters_over_time_min(
        df,
        time_col=time_col,
        label_col=cluster_col,
        time_unit=time_unit,
        bin_size_ms=bin_size_ms,
        sigma_bins=sigma_bins,
    )

    # --------------------------------------------------
    # C) Transition matrix
    # --------------------------------------------------
    print("→ Computing cluster transition matrix")
    trans_mat = cluster_transition_matrix(df, cluster_col)

    plt.figure(figsize=(5.5, 4.5))
    order = list(trans_mat.index)
    sns.heatmap(
        trans_mat.loc[order, order],
        annot=True,
        cmap=heatmap_cmap,
        cbar=False,
    )
    plt.title(f"{cluster_col} transition probabilities")
    plt.xlabel("Next spike cluster")
    plt.ylabel("Current spike cluster")
    plt.tight_layout()
    plt.show()

    
    # --------------------------------------------------
    # D) High vs Low feature distribution diagnostics
    # --------------------------------------------------
    print("→ Visualizing high vs low feature distributions")
    
    visualize_feature_groups_hist(
        df,
        features=SPIKE_WAVEFORM_FEATURES,
        group_col=cluster_col,
        groups=("high", "low"),
        group_names=("High", "Low"),
    )


    print("===== DONE =====\n")

    return {
        "transition_matrix": trans_mat,
    }


def visualize_feature_groups_hist(
    df: pd.DataFrame,
    features: list,
    group_col: str,
    groups: tuple = ("high", "low"),
    group_names: Optional[Tuple] = None,
    bins: int = 40,
):

    """
    Visualize feature distributions for two groups using histograms + QQ plots.


    Parameters
    ----------
    df : pd.DataFrame
    features : list[str]
        Feature names to visualize.
    group_col : str
        Column defining the two groups (e.g. cluster column).
    groups : tuple(str, str)
        Group labels to compare (default: ("high","low")).
    group_names : tuple(str, str) or None
        Pretty names for plotting. Defaults to group labels.
    """

    g1, g2 = groups
    if group_names is None:
        name1, name2 = g1, g2
    else:
        name1, name2 = group_names

    for feat in features:

        a = pd.to_numeric(df.loc[df[group_col] == g1, feat], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[df[group_col] == g2, feat], errors="coerce").dropna()

        if len(a) < 5 or len(b) < 5:
            continue

        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        fig.suptitle(f"{feat}", fontsize=12, weight="bold")

        # -------------------------
        # Histogram (overlayed)
        # -------------------------
        color_map = {
        "high": "#ff7f0e",  # orange
        "low":  "#1f77b4",  # blue
        }
    
        c1 = color_map.get(g1, "#4c72b0")
        c2 = color_map.get(g2, "#4c72b0")
        
        sns.histplot(
            a,
            ax=axes,
            color=c1,
            label=name1,
            stat="count",
            alpha=0.5,
            kde=False
        )
        
        sns.histplot(
            b,
            ax=axes,
            color=c2,
            label=name2,
            stat="count",
            alpha=0.5,
            kde=False
        
        )

        axes.set_title("Distribution")
        axes.legend()

       

        plt.tight_layout()
        plt.show()



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
    colors = [get_cluster_color(lab) for lab in labels]
    if plot:
        plt.figure(figsize=(7.5, 4.2))
        for i in range(k):
            plt.hist(data[assign_sorted == i], bins=bins, alpha=0.6, label=labels[i],color = colors[i])
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
    manual_thresholds: Optional[Dict[str, float]] = None,
):
    df_out = df.copy()
    if manual_thresholds is None:
        manual_thresholds = {}

    if features is None:
        features = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    report: Dict[str, Dict[str, Any]] = {}

    for feat in features:
        x = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x)
        data = x[valid]
        new_col = f"{feat}{suffix}"

        # ==========================================================
        # 1️⃣ MANUAL THRESHOLD — ALWAYS RUN FIRST
        # ==========================================================
        if feat in manual_thresholds:
            thr = manual_thresholds[feat]
            labels = ["low", "high"]

            cluster_assign = np.where(data <= thr, labels[0], labels[1])
            df_out.loc[valid, new_col] = cluster_assign
            df_out.loc[~valid, new_col] = np.nan

            report[feat] = {
                "k": 2,
                "centers": [np.nan, np.nan],
                "cutoffs": [thr],
                "diagnostics": {"manual_threshold": thr},
                "column": new_col,
                "labels": labels,
            }

            if plot_each:
                plt.figure(figsize=(6, 4))
                plt.hist(data[data <= thr], bins=10, alpha=0.7, label="low")
                plt.hist(data[data > thr], bins=10, alpha=0.7, label="high")
                plt.axvline(thr, color="k", linestyle="--", linewidth=2)
                plt.title(f"{feat} (manual threshold)")
                plt.legend()
                plt.show()

            continue  # 

        # ==========================================================
        # 2️⃣ HEURISTIC GATES (ONLY FOR AUTO MODE)
        # ==========================================================
        if np.unique(data).size < unique_min:
            continue

        k_suggest, diag = _looks_clustered_1d(
            data,
            min_peak_ratio=peak_ratio,
            max_valley_ratio=valley_ratio,
            max_k=max_k,
        )
        if k_suggest < 2:
            continue

        if labels_map and feat in labels_map:
            labels = labels_map[feat]
            if len(labels) != k_suggest:
                labels = ["low", "high"] if k_suggest == 2 else ["low", "mid", "high"][:k_suggest]
        else:
            labels = ["low", "high"] if k_suggest == 2 else ["low", "mid", "high"][:k_suggest]

        df_out, centers, cutoffs = kmeans_1d_cluster(
            df_out,
            feature=feat,
            k=k_suggest,
            labels=labels,
            new_col=new_col,
            plot=plot_each,
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
        ax_raster.scatter(
    t_sec[m], y,
    s=4,
    alpha=0.8,
    color=get_cluster_color(cl),
    label=str(cl)
)

    ax_raster.set_yticks(range(1, len(clusters) + 1))
    ax_raster.set_yticklabels(clusters)
    ax_raster.set_ylabel("cluster (raster)")
    ax_raster.legend(loc="upper right", fontsize=8)

    # rates
    for cl in clusters:
        m = (labels == cl)
        counts, _ = np.histogram(t_sec[m], bins=edges)
        rate = _smooth(counts / bin_w, sigma_bins)
        ax_rate.plot(
    centers, rate,
    label=str(cl),
    color=get_cluster_color(cl)
)

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
      next_spike_times_ms: list of next spike times (ms or NaN)
      meta_df: Dataframe describing each window
    """

    df = spk_df.copy()

    # Detect whether next-spike is required (ISI-based analyses only)
    requires_next_spike = "log_isi_cluster" in spk_df.columns

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
        if sid + 1 in df_sorted.index:
            t_next = float(df_sorted.loc[sid + 1, "spk_times_ms"])
        else:
            if requires_next_spike:
                continue          # ONLY drop for ISI-based analyses
            else:
                t_next = np.nan   # keep spike

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
            "gamma": (30, 55),
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





def run_time_resolved_specparam_per_spike(
    lfp_windows,        # list of windows, one per spike
    times_rel_list,     # same length
    next_rel_list,      # same length
    fs,
    **specparam_kwargs
):
    """
    Returns a list where index == spike index
    """
    all_results = []

    for i, (win, t_rel, next_rel) in enumerate(
        tqdm(zip(lfp_windows, times_rel_list, next_rel_list),
             total=len(lfp_windows),
             desc="Specparam per spike")
    ):
        out = run_time_resolved_specparam_on_window(
            lfp_window=win,
            times_rel=t_rel,
            fs=fs,
            next_spike_rel=next_rel,
            **specparam_kwargs
        )
        all_results.append(out)

    return all_results
    
def build_specparam_groups_from_spike_indices(
    specparam_by_spike,
    spike_indices_by_group,
):
    """
    spike_indices_by_group:
        {
          "Short ISI": [1, 5, 20, ...],
          "Long ISI":  [2, 7, 13, ...],
          ...
        }
    """
    grouped = {}

    for gname, inds in spike_indices_by_group.items():
        grouped[gname] = [specparam_by_spike[i] for i in inds]

    return grouped


# ------------------------------------------------------------------------------------------- #
# --------------------  Time-resolved and window visualziations  --------------------- #
# ------------------------------------------------------------------------------------------- #



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
                    "t_bins_s": 1D array,            # epoch center times in *seconds*
                    "offset": 1D array,
                    "exponent": 1D array,
                    "r_squared": 1D array,
                    "knee": 1D array or None,
                    "band_aucs": dict[band] -> 1D array,
                    "next_spike_rel": float or None, # usually relative time to next spike
                    "epoch_idx": 1D array (optional)
                },
                ...
            ]
        }
    feature : {"offset","exponent","r_squared","knee","band"}
        Which feature to extract.
    band : str, optional
        If feature == "band", which band key to pull from "band_aucs".

    Returns
    -------
    feat_groups : dict
        {
          group_name: {
             "windows":   list of 1D arrays of feature values,
             "times_rel": list of 1D arrays of times (in **seconds**),
             "next_rel":  1D array of next-spike times (in **seconds**, or NaN),
             "epoch_idx": list of epoch_idx arrays or None,
          }
        }
    """

    feat_groups = {}

    for group_name, win_list in time_res_results.items():

        windows_feat      = []
        windows_times     = []
        windows_next      = []
        windows_epoch_idx = []

        if not isinstance(win_list, (list, tuple)) or len(win_list) == 0:
            continue

        for w in win_list:

            # t_bins_s is already in SECONDS from run_time_resolved_specparam_on_window
            t_bins_s = np.asarray(w["t_bins_s"], float)

            # Handle next_spike_rel: try to keep it in seconds as well
            next_raw = w.get("next_spike_rel", None)
            if next_raw is None:
                next_rel_s = np.nan
            else:
                next_raw = float(next_raw)
                # Heuristic: if it's large, assume it was in ms and convert
                if np.abs(next_raw) > 20.0:
                    next_rel_s = next_raw / 1000.0   # ms -> s
                else:
                    next_rel_s = next_raw           # already seconds

            # ---------- PICK FEATURE ----------
            if feature in ["offset", "exponent", "r_squared", "knee"]:
                arr = w.get(feature, None)
                if arr is None:
                    continue

            elif feature == "band":
                if band is None:
                    raise ValueError("If feature=='band', you must pass band='gamma'/'theta'/etc.")
                band_aucs = w.get("band_aucs", {})
                arr = band_aucs.get(band, None)
                if arr is None:
                    continue

            else:
                raise ValueError(f"Unknown feature '{feature}'.")

            arr = np.asarray(arr, float)

            # safety: ensure array and time lengths match
            if arr.shape[0] != t_bins_s.shape[0]:
                # skip weird cases
                continue

            windows_feat.append(arr)
            windows_times.append(t_bins_s)          # *** seconds ***
            windows_next.append(next_rel_s)

            # optional: keep epoch indices if present
            if "epoch_idx" in w:
                windows_epoch_idx.append(np.asarray(w["epoch_idx"], int))
            else:
                windows_epoch_idx.append(None)

        if len(windows_feat) == 0:
            # no valid windows for this group
            continue

        feat_groups[group_name] = {
            "windows":   windows_feat,
            "times_rel": windows_times,                 # seconds
            "next_rel":  np.asarray(windows_next, float),  # seconds (or NaN)
            "epoch_idx": windows_epoch_idx,
        }

    return feat_groups









def plot_window_feature_groups_heatmap(
    groups: Dict[str, Dict[str, Any]],
    feature_label: str = "value",
    time_unit: str = "s",          # units of all groups' times_rel
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    sort_by: str = "next_rel",     # "next_rel" or "none"
    cmap: str = "viridis",
    titles: Optional[Dict[str, str]] = None,
    transition_time: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """
    For each group in `groups`, make a separate heatmap figure.

    Parameters
    ----------
    groups : dict
        {group_name: {"windows": [...], "times_rel": [...], "next_rel": optional}}
    feature_label : str
        Colorbar label.
    time_unit : {"s","ms"}
        Units of all groups' times_rel / next_rel.
    tmin, tmax : float or None
        Time window (in time_unit) to keep.
    sort_by : {"next_rel","none"}
        If "next_rel" and group has 'next_rel', events are sorted by next_rel and
        those values are used as orange "next spike" markers.
    cmap : str
        Colormap.
    titles : dict or None
        Optional mapping {group_name: title}; default: use group_name.
    transition_time : float
        Time of transition (same units as times_rel), vertical line & white markers.

    Returns
    -------
    results : dict
        {group_name: {"fig": fig, "ax": ax, "out": out_from_single}}
    """
    results: Dict[str, Dict[str, Any]] = {}

    for gname, g in groups.items():
        # decide sorting and markers for this group
        if sort_by == "next_rel" and "next_rel" in g:
            sort_vec   = np.asarray(g["next_rel"], float)
            next_times = sort_vec
        else:
            sort_vec   = None
            next_times = g.get("next_rel", None)

        # transition spike markers: usually all at transition_time
        n_events = len(g["windows"])
        trans_spike_times = np.full(n_events, transition_time, dtype=float)

        title = titles[gname] if (titles is not None and gname in titles) else gname

        fig, ax, out = plot_window_feature_group_heatmap_single(
            g,
            feature_label=feature_label,
            time_unit=time_unit,
            tmin=tmin,
            tmax=tmax,
            sort_by=sort_vec,
            title=title,
            cmap=cmap,
            transition_time=transition_time,
            trans_spike_times=trans_spike_times,
            next_spike_times=next_times,
            trans_label="Transition",
            trans_spike_label="Transition spike",
            next_spike_label="Next spike",
        )

        results[gname] = {"fig": fig, "ax": ax, "out": out}

    return results

def plot_window_feature_group_heatmap_single(
    group: Dict[str, Any],
    feature_label: str = "value",
    time_unit: str = "s",          # "s" or "ms" for group['times_rel']
    tmin: Optional[float] = None,  # in same units as times_rel
    tmax: Optional[float] = None,  # in same units as times_rel
    sort_by: Optional[Sequence[float]] = None,  # len = n_events, e.g. durations or next_rel
    title: Optional[str] = None,
    cmap: str = "viridis",
    # markers
    transition_time: float = 0.0,                  # vertical line position (same units as times_rel)
    trans_spike_times: Optional[Sequence[float]] = None,  # per-event, same units as times_rel
    next_spike_times: Optional[Sequence[float]] = None,   # per-event, same units as times_rel
    trans_label: str = "Transition",
    trans_spike_label: str = "Transition spike",
    next_spike_label: str = "Next spike",
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Plot a single group's windows as an event-by-time heatmap with optional spike markers.

    Parameters
    ----------
    group : dict
        Must have:
          - "windows"   : list of 1D arrays (feature values over time)
          - "times_rel" : list of 1D arrays (time for each window, same length as windows[i])
        Can have:
          - "next_rel"  : 1D array of next spike times (same units as times_rel)
    feature_label : str
        Label for the colorbar (e.g. "aperiodic_exponent", "gamma AUC", "LFP (µV)").
    time_unit : {"s","ms"}
        Unit of the values in times_rel (and marker times).
        X-axis is always seconds.
    tmin, tmax : float or None
        Time range to keep (same units as times_rel). If None, use full range.
    sort_by : array-like or None
        If given, used to sort events (rows). Must be length n_events.
        Common choice: the same as `next_spike_times` or event duration.
    title : str or None
        Figure title.
    cmap : str
        Matplotlib colormap.
    transition_time : float
        Time of the “transition” (e.g. 0), in same units as times_rel.
    trans_spike_times, next_spike_times : array-like or None
        Per-event times for the transition spike and next spike (same units as times_rel).

    Returns
    -------
    fig, ax, out
        out contains:
          - "Tgrid"   : 1D array of time in seconds
          - "A"       : 2D array (n_events, n_time)
          - "sort_idx": 1D array of row indices used
    """

    windows   = group["windows"]
    times_rel = group["times_rel"]

    if len(windows) == 0:
        raise ValueError("Group has no windows.")
    if len(windows) != len(times_rel):
        raise ValueError("len(windows) != len(times_rel).")

    n_events = len(windows)

    # ---------- base time grid ----------
    base_T = None
    for t in times_rel:
        t = np.asarray(t)
        if t.size > 1:
            base_T = t.astype(float)
            break
    if base_T is None:
        raise RuntimeError("No non-empty times_rel arrays found.")

    # crop to [tmin, tmax] in original units
    if tmin is not None or tmax is not None:
        if tmin is None:
            tmin = base_T.min()
        if tmax is None:
            tmax = base_T.max()
        mask = (base_T >= tmin) & (base_T <= tmax)
        base_T = base_T[mask]
    else:
        mask = slice(None)

    # convert grid to seconds
    if time_unit == "ms":
        Tgrid = base_T / 1000.0
        trans_time_sec = transition_time / 1000.0
    elif time_unit == "s":
        Tgrid = base_T.copy()
        trans_time_sec = float(transition_time)
    else:
        raise ValueError("time_unit must be 's' or 'ms'.")

    # ---------- build matrix A (events × time) ----------
    A_list = []
    for w, t in zip(windows, times_rel):
        w = np.asarray(w, float)
        t = np.asarray(t, float)
        if w.size != t.size or w.size < 2:
            A_list.append(np.full(base_T.shape, np.nan))
            continue

        t_crop = t[mask]
        w_crop = w[mask]
        if t_crop.size != base_T.size:
            yi = np.full(base_T.shape, np.nan, dtype=float)
            inside = (base_T >= t[0]) & (base_T <= t[-1])
            if inside.any():
                yi[inside] = np.interp(base_T[inside], t, w)
            A_list.append(yi)
        else:
            A_list.append(w_crop)
    A = np.vstack(A_list)  # (n_events, n_time)

    # ---------- sorting ----------
    sort_idx = np.arange(n_events)
    if sort_by is not None:
        sort_by = np.asarray(sort_by)
        if sort_by.shape[0] != n_events:
            raise ValueError("sort_by must have length n_events.")
        sort_idx = np.argsort(sort_by)
        A = A[sort_idx]
    
    # ---------- markers in seconds ----------
    def _prep_marker(times_arr, label: str):
        if times_arr is None:
            return None
        arr = np.asarray(times_arr, float)
        if arr.shape[0] != n_events:
            raise ValueError(f"{label} must have length n_events.")
        arr = arr[sort_idx]
        if time_unit == "ms":
            arr = arr / 1000.0
        return arr
    
    trans_spike_sec = _prep_marker(trans_spike_times, "trans_spike_times")
    next_spike_sec  = _prep_marker(next_spike_times, "next_spike_times")
    
    # ---------- plotting ----------
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(
        A,
        aspect="auto",
        origin="lower",
        extent=[Tgrid[0], Tgrid[-1], 0, n_events],
        cmap=cmap,
    )
    
    # vertical line at transition
    ax.axvline(trans_time_sec, color="white", linewidth=2, alpha=0.9, label=trans_label)
    
    # scatter markers: one per row
    y_rows = np.arange(n_events) + 0.5
    
    if trans_spike_sec is not None:
        ax.scatter(
            trans_spike_sec, y_rows,
            s=20, facecolors="none", edgecolors="white",
            linewidths=1.0, label=trans_spike_label, zorder=3,
        )
    
    if next_spike_sec is not None:
        ax.scatter(
            next_spike_sec, y_rows,
            s=20, facecolors="orange", edgecolors="black",
            linewidths=0.5, label=next_spike_label, zorder=3,
        )
    
    ax.set_xlabel("Time (s, relative to transition)")
    ax.set_ylabel("Events (sorted)" if sort_by is not None else "Events")
    if title is not None:
        ax.set_title(title)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(feature_label)
    
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(
            frameon=True, facecolor="black", edgecolor="none",
            framealpha=0.7, loc="upper right",
        )
    
    fig.tight_layout()
    
    out = {"Tgrid": Tgrid, "A": A, "sort_idx": sort_idx}
    return fig, ax, out
    

# ------------------------------------------------------------------------------------------- #
# ------------------------------ Functions and wrapper functions for post specparam cluster group analysis --------------------- #
# ------------------------------------------------------------------------------------------- #
def build_lfp_groups_from_clusters(
    df_features,
    all_spike_extractor,
    cluster_cols=None,
):
    """
    Build LFP groups from cluster columns in df_features.

    Returns:
        groups : dict
            groups[group_name] = {
                "windows":   list[np.ndarray],
                "times_rel": list[np.ndarray],
                "next_rel":  np.ndarray,
                "spk_inds":  np.ndarray,
            }
    """

    windows_all     = all_spike_extractor["windows"]
    t_rel_all       = all_spike_extractor["times_rel_ms"]
    spike_times_all = np.asarray(all_spike_extractor["spike_times_ms"], float)
    next_spk_all    = np.asarray(all_spike_extractor["next_spike_times_ms"], float)
    meta_df_all     = all_spike_extractor["meta_df"]

    # Map spk_id → index in windows_all
    spk_id_to_win = {
        sid: i for i, sid in enumerate(meta_df_all["spk_id"].values)
    }

    if cluster_cols is None:
        cluster_cols = [c for c in df_features.columns if c.endswith("_cluster")]

    groups = {}

    for col in cluster_cols:
        for label in df_features[col].dropna().unique():

            # spike IDs belonging to this cluster
            spk_ids = df_features.loc[df_features[col] == label, "spk_id"].values

            # map to window indices (only keep ones that exist)
            win_inds = [
                spk_id_to_win[sid]
                for sid in spk_ids
                if sid in spk_id_to_win
            ]

            if len(win_inds) == 0:
                continue

            group_name = f"{col}: {label}"

            groups[group_name] = {
                "windows":   [windows_all[i] for i in win_inds],
                "times_rel": [t_rel_all[i]   for i in win_inds],
                "next_rel":  next_spk_all[win_inds] - spike_times_all[win_inds],
                "spk_inds":  np.asarray(win_inds, dtype=int),
            }

    return groups


def make_specparam_feature_groups(
    specparam_by_spike,
    groups,
    feature: str,
    band: str = None,
    next_rel_unit: str = "ms",   # groups['next_rel'] is in ms in your current code
):
    """
    Build a groups-style dict (windows/times_rel/next_rel) for specparam features,
    using specparam_by_spike (computed once per spike) and groups (which store spk_inds).

    Returns:
        feat_groups[gname] = {
            "windows":   list of 1D arrays (feature over epochs),
            "times_rel": list of 1D arrays (epoch times, seconds),
            "next_rel":  1D array (seconds),
            "spk_inds":  np.ndarray,
        }
    """
    out = {}

    for gname, gdict in groups.items():
        spk_inds = np.asarray(gdict.get("spk_inds", []), dtype=int)
        if spk_inds.size == 0:
            continue

        win_list = []
        t_list   = []
        next_list = []

        # next_rel for these events (convert to seconds)
        next_rel = np.asarray(gdict.get("next_rel", np.full(spk_inds.size, np.nan)), float)
        if next_rel_unit == "ms":
            next_rel_s = next_rel / 1000.0
        else:
            next_rel_s = next_rel

        # collect per-spike traces
        for k, ind in enumerate(spk_inds):
            if ind < 0 or ind >= len(specparam_by_spike):
                continue

            res = specparam_by_spike[ind]
            if res is None:
                continue

            t_bins_s = np.asarray(res.get("t_bins_s", []), float)
            if t_bins_s.size == 0:
                continue

            if feature in ["offset", "exponent", "r_squared", "knee"]:
                arr = res.get(feature, None)
            elif feature == "band":
                if band is None:
                    raise ValueError("feature='band' requires band='gamma'/'alpha'/etc.")
                arr = res.get("band_aucs", {}).get(band, None)
            else:
                raise ValueError(f"Unknown feature: {feature}")

            if arr is None:
                continue

            arr = np.asarray(arr, float)
            if arr.size != t_bins_s.size:
                continue

            win_list.append(arr)
            t_list.append(t_bins_s)
            next_list.append(next_rel_s[k])

        if len(win_list) == 0:
            continue

        out[gname] = {
            "windows":   win_list,
            "times_rel": t_list,                       # seconds
            "next_rel":  np.asarray(next_list, float), # seconds
            "spk_inds":  spk_inds,
        }

    return out


def window_feature_group_traces_ci_delta(
    feat_groups,
    time_unit="s",
    ylabel="Δ Value",
    title=None,
    baseline_window=(-0.3, -0.05),
    analysis_window=(-0.2, 0.0),
    alpha_ci=0.25,
    plot=True,
    plot_mode="all",   # NEW: "all" or "per_cluster"
):
    """
    Plot mean ± 95% CI of baseline-corrected (Δ) feature traces.

    - Computation is unchanged
    - Plotting can be:
        * "all"         → one figure with all groups
        * "per_cluster" → one figure per cluster family

    Returns
    -------
    window_results : dict
        {group_name: list of per-epoch analysis-window means}
    """

    # ---------------------------
    # Organize groups by cluster
    # ---------------------------
    cluster_families = {}

    for gname in feat_groups.keys():
        # infer cluster family from name
        if ":" in gname:
            family = gname.split(":")[0]
        else:
            family = "all"

        cluster_families.setdefault(family, []).append(gname)

    # Decide which plotting sets to loop over
    if plot_mode == "all":
        plot_sets = {"All groups": list(feat_groups.keys())}
    elif plot_mode == "per_cluster":
        plot_sets = cluster_families
    else:
        raise ValueError("plot_mode must be 'all' or 'per_cluster'")

    window_results = {}

    # ==========================================================
    # LOOP OVER PLOTTING SETS (this is the ONLY new loop)
    # ==========================================================
    for set_name, group_names in plot_sets.items():

        if plot:
            fig, ax = plt.subplots(figsize=(12, 4))

        # -----------------------------------------
        # LOOP OVER GROUPS (unchanged computation)
        # -----------------------------------------
        for gname in group_names:

            g = feat_groups[gname]
            windows   = g["windows"]
            times_rel = g["times_rel"]
            color     = g.get("color", None)

            # --- determine common time grid ---
            base_T = None
            for t in times_rel:
                if len(t) > 1:
                    base_T = np.asarray(t, float)
                    break
            if base_T is None:
                continue

            Tgrid = base_T / 1000.0 if time_unit == "ms" else base_T.copy()

            mats = []
            epoch_win_means = []

            for w, t in zip(windows, times_rel):
                w = np.asarray(w, float)
                t = np.asarray(t, float)

                if w.size != t.size or w.size < 2:
                    continue

                t_sec = t / 1000.0 if time_unit == "ms" else t

                # ---- BASELINE SUBTRACTION ----
                b0, b1 = baseline_window
                bmask = (t_sec >= b0) & (t_sec <= b1)
                if not np.any(bmask):
                    continue

                baseline = np.nanmean(w[bmask])
                w = w - baseline

                # ---- interpolate onto common grid ----
                yi = np.full_like(Tgrid, np.nan)
                inside = (Tgrid >= t_sec[0]) & (Tgrid <= t_sec[-1])
                if inside.any():
                    yi[inside] = np.interp(Tgrid[inside], t_sec, w)
                    mats.append(yi)

                # ---- ANALYSIS WINDOW ----
                w0, w1 = analysis_window
                amask = (t_sec >= w0) & (t_sec <= w1)
                epoch_win_means.append(np.nanmean(w[amask]))

            if len(mats) < 2:
                continue

            A = np.vstack(mats)
            mean = np.nanmean(A, axis=0)
            std  = np.nanstd(A, axis=0)
            n    = np.sum(np.isfinite(A), axis=0)
            ci95 = 1.96 * std / np.sqrt(n)

            window_results[gname] = epoch_win_means

            if plot:
                ax.plot(Tgrid, mean, label=gname, color=color)
                ax.fill_between(
                    Tgrid,
                    mean - ci95,
                    mean + ci95,
                    color=color,
                    alpha=alpha_ci,
                    linewidth=0,
                )

        # ---------------------------
        # Finalize figure
        # ---------------------------
        if plot:
            ax.axvline(0, color="k", linestyle="--", linewidth=1)
            ax.set_xlabel("Time (s, relative to spike)")
            ax.set_ylabel(ylabel)

            if title:
                ax.set_title(f"{title} — {set_name}")
            else:
                ax.set_title(set_name)

            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
            )

            fig.tight_layout(rect=[0, 0, 0.82, 1])
            plt.show()

    return window_results




def stats_boxplot_from_window_results(
    window_results: dict,
    group_colors=None,
    plot_mode: str = "all",      # "all" or "per_cluster"
    feature_label: str = "Δ Value",
    figsize=(10, 5),
):
    """
    Boxplots + stats directly from precomputed per-epoch window values.
    """

    # -----------------------------
    # organize by cluster family
    # -----------------------------
    families = {}
    for g in window_results:
        fam = g.split(":")[0] if ":" in g else "all"
        families.setdefault(fam, []).append(g)

    if plot_mode == "all":
        plot_sets = {"All groups": list(window_results.keys())}
    elif plot_mode == "per_cluster":
        plot_sets = families
    else:
        raise ValueError("plot_mode must be 'all' or 'per_cluster'")

    report = {}

    default_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # -----------------------------
    # plotting + stats
    # -----------------------------
    for title, groups in plot_sets.items():

        data = [window_results[g] for g in groups if len(window_results[g]) > 0]
        labels = [g for g in groups if len(window_results[g]) > 0]

        if len(data) < 2:
            continue

        fig, ax = plt.subplots(figsize=figsize)

        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )

        # -----------------------------
        # COLOR LOGIC (ONLY CHANGE)
        # -----------------------------
        for i, box in enumerate(bp["boxes"]):

            if plot_mode == "per_cluster":
                lname = labels[i].lower()
                if "low" in lname:
                    col = "#1f77b4"   # blue
                elif "high" in lname:
                    col = "#ff7f0e"   # orange
                else:
                    col = "#7f7f7f"   # fallback
            else:
                # plot_mode == "all"
                col = default_palette[i % len(default_palette)]

            box.set_facecolor(col)
            box.set_alpha(0.6)

        # -----------------------------
        # stats
        # -----------------------------
        if len(data) == 2:
            from scipy.stats import ttest_ind
            stat, p = ttest_ind(data[0], data[1], equal_var=False)
            test = "t-test"
        else:
            from scipy.stats import f_oneway
            stat, p = f_oneway(*data)
            test = "anova"

        stars = p_to_stars(p)

        ax.set_title(f"{feature_label} — {title}")
        ax.set_ylabel(feature_label)

        ax.text(
            0.5, 0.92,
            f"{stars}  (p={p:.3g})",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.show()

        report[title] = {
            "groups": labels,
            "test": test,
            "p": p,
            "stars": stars,
        }

    return report




def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."
