"""
skp_feat_cluster_analysis.py - Comprehensive module for analysis of spe1 spike/lfp based on the clustering observed in the spike features 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
# ------------------------------------------------------------------------------------------- #
# ------------------------------ Cluster features that show groupped data --------------------
# ------------------------------------------------------------------------------------------- #

  

# ---- 1D K-means  ----
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

    # sort centers + remap labels
    order = np.argsort(centers)
    centers = centers[order]
    remap = {old_i: new_i for new_i, old_i in enumerate(order)}
    assign_sorted = np.vectorize(remap.get)(assign)
    cutoffs = (centers[:-1] + centers[1:]) / 2.0

    # write labels back at original indices
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
        plt.xlabel(feature); plt.ylabel("count")
        plt.title(f"{feature} clusters (k={k})"); 
        if k <= 10: plt.legend()
        plt.tight_layout(); plt.show()

    return df_out, centers, cutoffs


# ---- simple multimodality detector (histogram peaks) ----
def _fd_bins(data: np.ndarray) -> int:
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
    # gentle smoothing to reduce spurious wiggles
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    sm = np.convolve(hist, kernel, mode='same')
    # peaks = points greater than immediate neighbors
    peaks = np.where((sm[1:-1] > sm[:-2]) & (sm[1:-1] > sm[2:]))[0] + 1
    return len(peaks), peaks

def _looks_clustered_1d(
    data: np.ndarray,
    min_peak_ratio: float = 0.10,   # smallest acceptable peak height relative to max peak
    max_valley_ratio: float = 0.70, # valley height / min(peak1,peak2) must be below this
    max_k: int = 3
) -> Tuple[int, Dict[str, Any]]:
    """Return suggested k (1..max_k) and diagnostics."""
    if data.size < 50:  # small samples can be noisy; still try but be conservative
        pass
    bins = _fd_bins(data)
    hist, edges = np.histogram(data, bins=bins)
    if hist.max() == 0:
        return 1, {"reason": "empty hist"}
    # normalize
    h = hist.astype(float) / hist.max()
    n_peaks, peak_idx = _count_peaks_smooth(h)
    if n_peaks < 2:
        return 1, {"peaks": n_peaks}

    # take the two tallest peaks
    peak_heights = h[peak_idx]
    top2 = np.argsort(peak_heights)[-2:]
    p_idx = peak_idx[top2]
    p_idx.sort()
    p1, p2 = p_idx[0], p_idx[1]
    h1, h2 = h[p1], h[p2]

    # filter out tiny secondary bumps
    if min(h1, h2) < min_peak_ratio:
        return 1, {"reason": "secondary peak too small", "h1": h1, "h2": h2}

    # valley between them
    valley = h[(p1+1):p2].min() if p2 > p1 + 1 else min(h1, h2)
    valley_ratio = valley / min(h1, h2) if min(h1, h2) > 0 else 1.0
    if valley_ratio >= max_valley_ratio:
        return 1, {"reason": "valley too shallow", "valley_ratio": valley_ratio}

    # if we got here, at least bimodal. allow k=3 if a third peak exists and is meaningful
    k_suggest = 2
    if max_k >= 3 and n_peaks >= 3:
        # third peak: ensure it’s not tiny and separated
        third_idx = [i for i in peak_idx if i not in (p1, p2)]
        third_heights = [h[i] for i in third_idx]
        if len(third_heights) > 0 and max(third_heights) >= min_peak_ratio:
            k_suggest = 3

    # diag info
    centers_approx = (edges[:-1] + edges[1:]) / 2.0
    approx_peaks_vals = centers_approx[peak_idx]
    return k_suggest, {
        "peaks": int(n_peaks),
        "peak_bins": peak_idx,
        "peak_vals_approx": approx_peaks_vals.tolist(),
        "valley_ratio": float(valley_ratio)
    }







# ---- main: scan features, detect clustering, and apply 1D K-means ----
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

    Parameters
    ----------
    df : DataFrame
    features : list[str] or None
        Which columns to consider. If None, all numeric columns.
    max_k : int
        Maximum clusters to consider (2 or 3 recommended).
    labels_map : dict[str, list[str]] or None
        Optional mapping: feature -> list of labels to use (must match chosen k).
        If None, uses generic ['low','high'] or ['low','mid','high'].
    suffix : str
        Suffix for new cluster columns: f"{feature}{suffix}".
    plot_each : bool
        If True, show per-feature clustering plot.
    peak_ratio : float
        Minimum height (relative to max peak) to consider a peak “real”.
    valley_ratio : float
        Maximum allowed valley/peak ratio; lower means stronger separation.
    unique_min : int
        Skip features with < unique_min valid values.

    Returns
    -------
    df_out : DataFrame
        Copy of df with added cluster columns (only for detected features).
    report : dict
        feature -> { 'k': int, 'centers': np.ndarray, 'cutoffs': np.ndarray, 'diagnostics': {...} }
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
        if data.size < 50:
            # still allowed, but the detector is more conservative with small N
            pass

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
                # if mismatch, truncate or extend generically
                default = ['low','high'] if k_suggest == 2 else ['low','mid','high'][:k_suggest]
                labels = default
        else:
            labels = ['low','high'] if k_suggest == 2 else ['low','mid','high'][:k_suggest]

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
