"""
stim_artifact_check.py
-----------------------
Checks for signatures of experimenter-controlled stimulation in the spe-1
patch-clamp traces (co-author concern: imperfect bridge balance during a
current step could look like a membrane-potential shift or a brief
capacitance transient rather than genuine spontaneous activity).

Three checks, run per cell:
  1. Waveform kinetics  - do any threshold-crossing events decay like a slow
     (~1/(2*pi*10 Hz) ~= 16 ms tau) RC/bridge-balance transient instead of a
     fast, multiphasic AP? (`event_shape_features`, `detect_events_with_shape`)
  2. Sub-threshold transients - repeat (1) at a lower amplitude threshold to
     catch weaker deflections that would not have been picked up as spikes.
  3. Timing regularity - do event times show a periodic/regular structure
     (ISI peakiness, narrowband PSD peak) inconsistent with spontaneous,
     irregular firing? (`isi_regularity`, `psd_peakiness`)

All functions operate on the already-filtered patch traces produced by
`data_loader.load_spe1_data` (bandpass 10-25000 Hz), since raw unfiltered
binaries are not retained on disk.

Reuse notes (what was and wasn't reused from the existing pipeline, and why):
  - Spike *detection* reuses `spikewise.patch.window.find_spike_times` /
    `peak_distance_to_samples` directly (same detector the paper's `Spike`
    class uses), so results are directly comparable to the paper's pipeline.
  - `spikewise.patch.points.control_points` + `compute_peak_features` compute
    the paper's own `peak_width` feature, but were measured (on c18's
    168,904-event sub-threshold pass) at ~64 ms/event - about 3 hours for one
    cell's sub-threshold check alone. That's not tractable across 43 cells, so
    `event_shape_features` below is a fast, vectorized stand-in for the same
    "how wide is this event" question, not a reimplementation for its own sake.
  - The cached `cluster_pickles/c{id}_cluster_df.pkl` per-spike features were
    considered as a free source of `peak_width` at the spike threshold, but
    they hold a curated subset after `Spike.filter_features()` (e.g. c18: 8
    cached rows vs. 23,304 raw threshold-crossing events) - exactly the kind
    of poor-fitting/atypical event this check is looking for could already be
    filtered out, so they're not used here.
  - ISI computation reuses `spikewise.patch.features.compute_isi`.
  - Per-cell summary caching reuses `spk_feat_cluster_analysis.load_or_compute`.
"""

import os
import pickle
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.signal import welch, find_peaks, peak_widths

try:
    from .config import (DICT_PATCH_FS, DICT_SPK_THRESH, DICT_PATCH_TYPE,
                          SPE1_DATA_ROOT, SPE1_PICKLE_ROOT)
    from .spk_feat_cluster_analysis import load_or_compute
except ImportError:
    from config import (DICT_PATCH_FS, DICT_SPK_THRESH, DICT_PATCH_TYPE,
                         SPE1_DATA_ROOT, SPE1_PICKLE_ROOT)
    from spk_feat_cluster_analysis import load_or_compute

from spikewise.patch.window import find_spike_times, peak_distance_to_samples
from spikewise.patch.features import compute_isi


PATCH_DIR_DEFAULT = os.path.join(SPE1_DATA_ROOT, "filt_patch_recordings")

# RC time constant implied by the 10 Hz highpass edge of the patch filter,
# in ms. A step (e.g. a bridge-imbalance offset switching on) passed through
# this filter decays with this tau; a genuine AP does not.
HIGHPASS_TAU_MS = 1000.0 / (2 * np.pi * 10.0)  # ~15.9 ms


def load_patch_trace(cell_num: int, patch_dir: str = PATCH_DIR_DEFAULT):
    """Load a cell's filtered patch trace and its sampling rate.

    Returns
    -------
    trace : 1d array (mV, bandpass filtered 10-25000 Hz)
    fs : float (Hz)
    """
    fs = DICT_PATCH_FS[cell_num]
    trace = np.load(os.path.join(patch_dir, f"c{cell_num}_patch.npy"))
    return trace, fs


def event_shape_features(trace: np.ndarray, idx: int, fs: float,
                          pre_ms: float = 10.0, post_ms: float = 10.0,
                          decay_check_ms: float = 8.0):
    """Shape/kinetics features for one threshold-crossing event.

    A genuine AP is fast and multiphasic (rise, peak, fast repolarization,
    often an undershoot), so little of the peak amplitude should remain a
    few ms after the peak and the derivative should change sign more than
    once on the way down. A bridge-balance / capacitance-transient artifact
    riding on the 10 Hz highpass edge decays monotonically with a tau of
    ~16 ms, so most of the peak amplitude is still present after
    `decay_check_ms` and the decay is monotonic.

    Returns
    -------
    dict with: fwhm_ms, post_frac, n_sign_changes_post, peak_amp
    """
    n_pre = int(round(pre_ms * fs / 1000))
    n_post = int(round(post_ms * fs / 1000))
    lo, hi = idx - n_pre, idx + n_post
    if lo < 0 or hi >= len(trace):
        return None

    window = trace[lo:hi + 1]
    local_peak = n_pre  # index of the event within the window
    peak_amp = window[local_peak]
    baseline = np.median(np.concatenate([window[:max(1, n_pre // 4)],
                                          window[-max(1, n_post // 4):]]))

    half_max = baseline + (peak_amp - baseline) / 2.0
    pre_seg = window[:local_peak + 1]
    post_seg = window[local_peak:]

    below_pre = np.where(pre_seg <= half_max)[0]
    below_post = np.where(post_seg <= half_max)[0]
    if len(below_pre) == 0 or len(below_post) == 0:
        fwhm_ms = np.nan
    else:
        rise_i = below_pre[-1]
        fall_i = local_peak + below_post[0]
        fwhm_ms = (fall_i - rise_i) * 1000.0 / fs

    n_decay = int(round(decay_check_ms * fs / 1000))
    n_decay = min(n_decay, len(post_seg) - 1)
    val_at_decay = post_seg[n_decay]
    post_frac = (val_at_decay - baseline) / (peak_amp - baseline + 1e-12)

    d_post = np.diff(post_seg[:n_decay + 1])
    sign_changes = int(np.sum(np.diff(np.sign(d_post)) != 0))

    return {
        "fwhm_ms": fwhm_ms,
        "post_frac": post_frac,
        "n_sign_changes_post": sign_changes,
        "peak_amp": peak_amp,
    }


def detect_events_with_shape(trace: np.ndarray, fs: float, thresh_amp: float,
                              thresh_ms: float = 1.0,
                              post_frac_cutoff: float = 0.3,
                              max_sign_changes: int = 1):
    """Detect threshold-crossing events and flag ones with artifact-like kinetics.

    Uses the same peak detector as the paper's spike pipeline
    (`spikewise.patch.window.find_spike_times`), so `events` reproduces the
    spikes that pipeline would extract at `thresh_amp`.

    A candidate artifact is an event whose shape is monotonic and slow-decaying
    (`post_frac >= post_frac_cutoff` and `n_sign_changes_post <= max_sign_changes`)
    rather than fast and multiphasic like a real AP.

    Returns
    -------
    events : dict of arrays - idx, time_ms, peak_amp, fwhm_ms, post_frac,
             n_sign_changes_post, is_candidate_artifact
    """
    distance = peak_distance_to_samples(thresh_ms, fs)
    idx_events, amp_events = find_spike_times(trace, thresh_amp, distance)

    rows = []
    for idx, amp in zip(idx_events, amp_events):
        feats = event_shape_features(trace, idx, fs)
        if feats is None:
            continue
        is_artifact = (feats["post_frac"] >= post_frac_cutoff and
                        feats["n_sign_changes_post"] <= max_sign_changes)
        rows.append({
            "idx": idx,
            "time_ms": idx * 1000.0 / fs,
            **feats,
            "is_candidate_artifact": is_artifact,
        })

    if not rows:
        return {k: np.array([]) for k in
                ["idx", "time_ms", "peak_amp", "fwhm_ms", "post_frac",
                 "n_sign_changes_post", "is_candidate_artifact"]}

    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def isi_regularity(spike_inds: np.ndarray, fs: float, n_bins: int = 50):
    """Coefficient of variation and log-ISI histogram peakiness.

    ISIs are computed with `spikewise.patch.features.compute_isi` (the same
    inter-spike-interval function used elsewhere in the pipeline).

    Spontaneous firing has irregular ISIs (CV not near 0, no single dominant
    ISI mode). A fixed-interval stimulation protocol produces a narrow ISI
    distribution concentrated in one or a few bins. ISI histograms are built
    in log-space (standard for spike trains, which are heavy-tailed/bursty)
    so that a genuine narrow periodic mode isn't masked by, nor confused
    with, the pileup of short ISIs a linear-binned histogram gets from bursts.

    Returns
    -------
    dict with: n_events, isi_cv, isi_log_modal_frac (fraction of ISIs in the
    single most populated log-spaced bin)
    """
    n_events = len(spike_inds)
    if n_events < 3:
        return {"n_events": n_events, "isi_cv": np.nan, "isi_log_modal_frac": np.nan}

    isi = compute_isi(np.sort(spike_inds), fs, in_ms=True)
    isi = isi[~np.isnan(isi)]
    isi = isi[isi > 0]
    if len(isi) < 2:
        return {"n_events": n_events, "isi_cv": np.nan, "isi_log_modal_frac": np.nan}

    cv = np.std(isi) / (np.mean(isi) + 1e-12)
    log_isi = np.log10(isi)
    counts, _ = np.histogram(log_isi, bins=n_bins)
    modal_frac = counts.max() / counts.sum()

    return {"n_events": n_events, "isi_cv": cv, "isi_log_modal_frac": modal_frac}


def psd_peakiness(trace: np.ndarray, fs: float, fmax: float = 100.0,
                   nperseg: int = 2 ** 16,
                   exclude_bands=((47, 53), (97, 103), (147, 153)),
                   narrowband_cutoff_hz: float = 3.0):
    """Welch PSD peakiness below `fmax`, excluding mains-noise harmonics.

    Finds the most prominent local peak riding on top of the aperiodic
    (1/f-like) background via `scipy.signal.find_peaks` on the log-power
    spectrum, where prominence is relative to each peak's local surrounding
    valleys rather than a global reference (this avoids simply flagging the
    low-frequency end of a normal 1/f spectrum, which always has the
    highest absolute power).

    Critically, a genuine discrete/periodic stimulation artifact (e.g. a
    repeated test/current pulse) shows up as a narrow spectral *line*
    whose width is set by the recording duration, i.e. close to the Welch
    frequency resolution - not a broad bump. `is_narrowband_peak` filters
    on `peak_fwhm_hz < narrowband_cutoff_hz` for this reason: a broad
    low-frequency bump (e.g. from whole-cell subthreshold Vm fluctuations)
    would otherwise be indistinguishable from a real periodicity by
    prominence alone.

    Returns
    -------
    dict with: peak_freq_hz, peak_prominence (fold-change in linear power
    at the peak vs. its local valley), peak_fwhm_hz, is_narrowband_peak,
    freq_resolution_hz
    """
    freqs, pxx = welch(trace, fs=fs, nperseg=min(nperseg, len(trace)))
    sel = freqs <= fmax
    freqs, pxx = freqs[sel], pxx[sel]
    freq_res = freqs[1] - freqs[0]

    mask = np.ones_like(freqs, dtype=bool)
    for lo, hi in exclude_bands:
        mask &= ~((freqs >= lo) & (freqs <= hi))
    freqs, pxx = freqs[mask], pxx[mask]

    if len(freqs) < 10:
        return {"peak_freq_hz": np.nan, "peak_prominence": np.nan,
                "peak_fwhm_hz": np.nan, "is_narrowband_peak": False,
                "freq_resolution_hz": freq_res}

    log_pxx = np.log10(pxx + 1e-20)
    peak_idxs, props = find_peaks(log_pxx, prominence=0)
    if len(peak_idxs) == 0:
        return {"peak_freq_hz": np.nan, "peak_prominence": 0.0,
                "peak_fwhm_hz": np.nan, "is_narrowband_peak": False,
                "freq_resolution_hz": freq_res}

    best = np.argmax(props["prominences"])
    peak_i = peak_idxs[best]
    prominence_log = props["prominences"][best]
    fwhm_bins = peak_widths(log_pxx, [peak_i], rel_height=0.5)[0][0]
    fwhm_hz = float(fwhm_bins * freq_res)

    return {
        "peak_freq_hz": float(freqs[peak_i]),
        "peak_prominence": float(10 ** prominence_log),
        "peak_fwhm_hz": fwhm_hz,
        "is_narrowband_peak": fwhm_hz < narrowband_cutoff_hz,
        "freq_resolution_hz": freq_res,
    }


def summarize_cell(cell_num: int, patch_dir: str = PATCH_DIR_DEFAULT,
                    low_thresh_frac: float = 0.4):
    """Run all three checks for one cell and return a flat summary dict."""
    trace, fs = load_patch_trace(cell_num, patch_dir)
    thresh = DICT_SPK_THRESH[cell_num]
    patch_type = DICT_PATCH_TYPE.get(cell_num, "unknown")
    current_type = patch_type.split(",")[-1].strip() if "," in patch_type else "unknown"

    spikes = detect_events_with_shape(trace, fs, thresh_amp=thresh)
    sub = detect_events_with_shape(trace, fs, thresh_amp=thresh * low_thresh_frac)

    isi_spk = isi_regularity(spikes["idx"], fs)
    psd = psd_peakiness(trace, fs)

    n_spk = len(spikes["idx"])
    n_spk_artifact = int(np.sum(spikes["is_candidate_artifact"])) if n_spk else 0
    n_sub = len(sub["idx"])
    n_sub_artifact = int(np.sum(sub["is_candidate_artifact"])) if n_sub else 0

    return {
        "cell_id": f"c{cell_num}",
        "current_type": current_type,
        "n_spikes": n_spk,
        "n_spikes_artifact_shape": n_spk_artifact,
        "frac_spikes_artifact_shape": n_spk_artifact / n_spk if n_spk else np.nan,
        "n_subthresh_events": n_sub,
        "n_subthresh_artifact_shape": n_sub_artifact,
        "frac_subthresh_artifact_shape": n_sub_artifact / n_sub if n_sub else np.nan,
        "isi_cv": isi_spk["isi_cv"],
        "isi_log_modal_frac": isi_spk["isi_log_modal_frac"],
        "psd_peak_freq_hz": psd["peak_freq_hz"],
        "psd_peak_prominence": psd["peak_prominence"],
        "psd_peak_fwhm_hz": psd["peak_fwhm_hz"],
        "psd_is_narrowband_peak": psd["is_narrowband_peak"],
    }


def run_all_cells(cell_nums, patch_dir: str = PATCH_DIR_DEFAULT, verbose: bool = True,
                   cache_path: str = None, force: bool = False):
    """Run `summarize_cell` across a list of cells; returns a pandas DataFrame.

    Caching (when `cache_path` is given) reuses
    `spk_feat_cluster_analysis.load_or_compute`, the same pickle-cache
    helper used by the rest of the spe-1 pipeline.
    """
    import pandas as pd
    from tqdm.auto import tqdm

    def _compute():
        rows = []
        iterator = tqdm(cell_nums) if verbose else cell_nums
        for cell_num in iterator:
            try:
                rows.append(summarize_cell(cell_num, patch_dir))
            except Exception as e:
                if verbose:
                    print(f"  [skip] c{cell_num}: {e}")
        return pd.DataFrame(rows)

    if cache_path is None:
        return _compute()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    return load_or_compute(cache_path, _compute, force=force, verbose=verbose)


# ------------------------------------------------------------------------------------------- #
# --------------------------- Raw baseline / changepoint checks ------------------------------ #
# ------------------------------------------------------------------------------------------- #
# A second, independent way of asking the same question as the checks above: instead of
# looking for artifact-shaped *events*, look for a real shift in the resting membrane
# potential itself, and compare its timing to when a cell's spike-waveform features shift
# cluster. Used by supp_stimulation_artifact_check.ipynb.

RAW_PATCH_DIR_DEFAULT = os.path.join(SPE1_DATA_ROOT, "raw_patch_recordings")
CLUSTER_PKL_DIR_DEFAULT = os.path.join(SPE1_PICKLE_ROOT, "cluster_pickles")
ORDINAL_MAP = {"low": 0, "mid": 1, "high": 2}

# waveform-SHAPE features only, not spike-timing/ISI features (log_isi, n_spikes_*) or the
# spike time/index itself, which get clustered too but aren't "waveform"
WAVEFORM_FEATURES = ["ramp_amp", "inflection_time", "inflection_amp",
                      "peak_amp", "peak_width", "peak_sharpness",
                      "exp_lambda", "exp_const"]


def load_raw_patch(cell_num: int, raw_dir: str = RAW_PATCH_DIR_DEFAULT):
    """Load a cell's raw, unfiltered patch recording (float64, in the units it was recorded in).

    Unlike `load_patch_trace` above (bandpass filtered), this is the true pre-processing
    recording, needed because filtering can hide a slow baseline shift.
    """
    fs = DICT_PATCH_FS[cell_num]
    trace = np.fromfile(os.path.join(raw_dir, f"c{cell_num}_patch_ch1.bin"), dtype="float64")
    return trace, fs


def compute_baseline(cell_num: int, bin_s: float = 1.0, raw_dir: str = RAW_PATCH_DIR_DEFAULT):
    """Median voltage per bin_s-second bin: the resting level with spikes averaged out."""
    trace, fs = load_raw_patch(cell_num, raw_dir)
    bin_samples = int(bin_s * fs)
    n_bins = len(trace) // bin_samples
    chunks = trace[:n_bins * bin_samples].reshape(n_bins, bin_samples)
    baseline = np.median(chunks, axis=1)
    t_min = np.arange(n_bins) / 60.0
    return t_min, baseline


def get_pelt_changepoints(signal: np.ndarray, t_min: np.ndarray):
    """PELT (L2 cost, BIC-style penalty) changepoints on a 1D signal already binned to match t_min."""
    valid = ~np.isnan(signal)
    if valid.sum() < 20 or np.nanstd(signal) < 1e-9:
        return []
    algo = rpt.Pelt(model="l2", min_size=5, jump=1).fit(signal.reshape(-1, 1))
    pen = 3 * np.log(len(signal)) * np.nanvar(signal)
    return [round(t_min[b], 2) for b in algo.predict(pen=pen) if b < len(signal)]


def bin_ordinal(spk_min: np.ndarray, ordinal: np.ndarray, n_bins: int, duration_min: float):
    """Per-spike ordinal cluster label (0/1/2) averaged into per-second bins, matching baseline bins."""
    bin_edges = np.arange(0, duration_min + 1 / 60, 1 / 60)[:n_bins + 1]
    bin_idx = np.clip(np.digitize(spk_min, bin_edges) - 1, 0, n_bins - 1)
    sums = np.bincount(bin_idx, weights=ordinal, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        binned = sums / counts
    binned[counts == 0] = np.nan
    return pd.Series(binned).interpolate(limit_direction="both").to_numpy()


def load_cluster_df(cell_num: int, cluster_dir: str = CLUSTER_PKL_DIR_DEFAULT):
    """A cell's per-spike waveform-feature dataframe, or None if it was never clustered."""
    path = os.path.join(cluster_dir, f"c{cell_num}_cluster_df.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def waveform_cluster_columns(df, features=WAVEFORM_FEATURES):
    """Which of `features` actually got clustered for this cell (some are unimodal, no _cluster column)."""
    return [f"{feat}_cluster" for feat in features if f"{feat}_cluster" in df.columns]


def strip_cluster(name):
    """Display name for a *_cluster column: drop the suffix, underscores to spaces."""
    return name.replace("_cluster", "").replace("_", " ")


def compute_all_baseline_changepoints(cell_nums, baselines):
    """PELT changepoints on every cell's baseline; skips cells with a degenerate (all-zero-step) baseline."""
    out = {}
    for cell_num in cell_nums:
        t_min, baseline = baselines[cell_num]
        typical_step = np.median(np.abs(np.diff(baseline)))
        if typical_step < 1e-9 or len(baseline) < 20:
            continue
        out[f"c{cell_num}"] = get_pelt_changepoints(baseline, t_min)
    return out


def compute_all_cluster_changepoints(cell_nums, baselines):
    """PELT changepoints on every clustered waveform-shape feature, for every cell."""
    out = {}   # cell_id -> {feature: [changepoint minutes]}
    for cell_num in cell_nums:
        df = load_cluster_df(cell_num)
        if df is None:
            continue
        feats = waveform_cluster_columns(df)
        spk_min = df["spk_times_ms"].to_numpy() / 60000.0
        t_min, baseline = baselines[cell_num]
        feat_cps = {}
        for feat in feats:
            ordinal = df[feat].map(ORDINAL_MAP).to_numpy().astype(float)
            signal = bin_ordinal(spk_min, ordinal, len(baseline), t_min[-1])
            cps = get_pelt_changepoints(signal, t_min)
            if cps:
                feat_cps[feat] = cps
        if feat_cps:
            out[f"c{cell_num}"] = feat_cps
    return out


def cluster_mix_after(df, feat, cp):
    """Cluster proportions among spikes after time cp (minutes); None if too few spikes to judge."""
    spk_min = df["spk_times_ms"].to_numpy() / 60000.0
    after = df[feat].to_numpy()[spk_min >= cp]
    if len(after) < 10:
        return None
    return pd.Series(after).value_counts(normalize=True)


# ------------------------------------------------------------------------------------------- #
# --------------------------------------- Plotting ------------------------------------------- #
# ------------------------------------------------------------------------------------------- #

def plot_cell_psd(cell_num: int, patch_dir: str = PATCH_DIR_DEFAULT, fmax: float = 100.0,
                   ax=None):
    """Plot a cell's Welch PSD with the detected peak and its FWHM span annotated.

    Lets a reader judge by eye whether the flagged peak is a narrow line
    (candidate periodic artifact) or a broad bump (aperiodic/physiological).
    """
    import matplotlib.pyplot as plt

    trace, fs = load_patch_trace(cell_num, patch_dir)
    psd = psd_peakiness(trace, fs, fmax=fmax)
    freqs, pxx = welch(trace, fs=fs, nperseg=min(2 ** 16, len(trace)))
    sel = freqs <= fmax

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.semilogy(freqs[sel], pxx[sel], color="black", lw=1)
    if np.isfinite(psd["peak_freq_hz"]):
        ax.axvline(psd["peak_freq_hz"], color="crimson", ls="--", lw=1,
                    label=f"peak {psd['peak_freq_hz']:.1f} Hz\n"
                          f"FWHM {psd['peak_fwhm_hz']:.1f} Hz, {psd['peak_prominence']:.1f}x\n"
                          f"{'narrowband' if psd['is_narrowband_peak'] else 'broad'}")
        half = psd["peak_fwhm_hz"] / 2
        ax.axvspan(psd["peak_freq_hz"] - half, psd["peak_freq_hz"] + half,
                    color="crimson", alpha=0.15)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(f"c{cell_num} patch PSD")
    ax.legend(fontsize=8, frameon=False)
    return ax


def plot_trace_zoom(cell_num: int, t_start_ms: float, t_end_ms: float,
                     patch_dir: str = PATCH_DIR_DEFAULT, ax=None):
    """Plot a raw patch-trace window with detected spikes and candidate
    artifact-shaped events marked, for visual inspection."""
    import matplotlib.pyplot as plt

    trace, fs = load_patch_trace(cell_num, patch_dir)
    thresh = DICT_SPK_THRESH[cell_num]
    lo, hi = int(t_start_ms * fs / 1000), int(t_end_ms * fs / 1000)
    seg = trace[lo:hi]
    t = np.arange(lo, hi) * 1000.0 / fs

    events = detect_events_with_shape(trace, fs, thresh_amp=thresh)
    in_win = (events["time_ms"] >= t_start_ms) & (events["time_ms"] <= t_end_ms)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    ax.plot(t, seg, color="black", lw=0.7)
    if np.any(in_win):
        real = in_win & ~events["is_candidate_artifact"]
        art = in_win & events["is_candidate_artifact"]
        ax.scatter(events["time_ms"][real], events["peak_amp"][real],
                    color="tab:blue", s=25, zorder=3, label="spike")
        if np.any(art):
            ax.scatter(events["time_ms"][art], events["peak_amp"][art],
                        color="crimson", s=35, marker="x", zorder=3,
                        label="candidate artifact shape")
    ax.axhline(thresh, color="gray", ls=":", lw=0.8, label="spike threshold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Patch (mV)")
    ax.set_title(f"c{cell_num} trace, {t_start_ms:.0f}-{t_end_ms:.0f} ms")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    return ax


def plot_summary_panels(df):
    """3-panel population summary: PSD peak prominence vs. FWHM (colored by
    narrowband flag), ISI CV by current type, and candidate-artifact-shape
    fraction per cell."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: PSD peak prominence vs FWHM
    ax = axes[0]
    colors = df["psd_is_narrowband_peak"].map({True: "crimson", False: "tab:blue"})
    ax.scatter(df["psd_peak_fwhm_hz"], df["psd_peak_prominence"], c=colors, s=40, alpha=0.8)
    ax.axvline(3.0, color="gray", ls="--", lw=1, label="narrowband cutoff (3 Hz)")
    ax.set_xlabel("PSD peak FWHM (Hz)")
    ax.set_ylabel("PSD peak prominence (fold)")
    ax.set_title("Peak width vs. prominence\n(red = narrowband candidate)")
    ax.legend(fontsize=8, frameon=False)

    # Panel 2: ISI CV by current type
    ax = axes[1]
    for i, ct in enumerate(sorted(df["current_type"].dropna().unique())):
        vals = df.loc[df["current_type"] == ct, "isi_cv"].dropna()
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.08, 0.08, len(vals)),
                   vals, alpha=0.7, s=30, label=ct)
    ax.set_xticks(range(len(df["current_type"].dropna().unique())))
    ax.set_xticklabels(sorted(df["current_type"].dropna().unique()))
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="CV = 1 (Poisson)")
    ax.set_ylabel("ISI coefficient of variation")
    ax.set_title("Firing irregularity by clamp mode\n(low CV would suggest fixed-interval stim)")
    ax.legend(fontsize=8, frameon=False)

    # Panel 3: candidate artifact-shape fraction
    ax = axes[2]
    order = df.sort_values("frac_subthresh_artifact_shape", ascending=False)
    x = np.arange(len(order))
    ax.bar(x - 0.2, order["frac_spikes_artifact_shape"], width=0.4, label="at spike threshold")
    ax.bar(x + 0.2, order["frac_subthresh_artifact_shape"], width=0.4, label="at 0.4x threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(order["cell_id"], rotation=90, fontsize=6)
    ax.set_ylabel("Fraction of events with\nartifact-like kinetics")
    ax.set_title("Slow, monotonic-decay events\nper cell")
    if (order["frac_spikes_artifact_shape"].fillna(0) == 0).all() and \
       (order["frac_subthresh_artifact_shape"].fillna(0) == 0).all():
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "0 candidate artifact-shaped\nevents in any cell",
                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="dimgray")
    ax.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    return fig, axes


CLUST_COLORS = {"low": "#0072B2", "mid": "#009E73", "high": "#D55E00"}  # Wong 2011 colorblind-safe


def plot_baseline_vs_cluster_example(cell_num: int, feature: str, split_time: float,
                                      split_label: str = "changepoint",
                                      cluster_dir: str = CLUSTER_PKL_DIR_DEFAULT):
    """Raw baseline, per-spike cluster raster, each cluster's average spike waveform, and a
    before/after cluster-mix panel, split at `split_time` (minutes). Used to visually compare
    a baseline event's timing against a waveform-cluster shift, whether or not the two coincide.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    t_min, baseline = compute_baseline(cell_num)
    df = load_cluster_df(cell_num)
    spk_min = df["spk_times_ms"].to_numpy() / 60000.0
    clusters = df[feature].to_numpy()

    before = pd.Series(clusters[spk_min < split_time]).value_counts(normalize=True)
    after = pd.Series(clusters[spk_min >= split_time]).value_counts(normalize=True)

    with open(os.path.join(cluster_dir, f"c{cell_num}_cluster_waveforms.pkl"), "rb") as f:
        wf = pickle.load(f)
    wf_feat = wf.get(feature, {})
    fs = DICT_PATCH_FS[cell_num]
    t_wave_ms = np.asarray(wf_feat["t_axis"]) / fs * 1000 if "t_axis" in wf_feat else None

    # constrained_layout (not tight_layout) is required here: tight_layout doesn't correctly
    # size a gridspec-spanning axis like ax_mix and silently corrupts/clips its text.
    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.8, 1.3], width_ratios=[5, 1.6],
                           wspace=0.05, hspace=0.15)
    ax_base = fig.add_subplot(gs[0, 0])
    ax_raster = fig.add_subplot(gs[1, 0], sharex=ax_base)
    ax_wave = fig.add_subplot(gs[2, 0])
    ax_mix = fig.add_subplot(gs[:, 1])

    feat_label = feature.replace("_cluster", "").replace("_", " ")
    ax_base.plot(t_min, baseline, color="black", lw=0.6)
    ax_base.set_ylabel("raw baseline\n(raw units, median/s)", fontsize=13)
    ax_base.set_title(f"c{cell_num}: raw baseline vs. {feat_label} over time", fontsize=15)
    ax_base.axvline(split_time, color="crimson", ls="--", lw=1, label=split_label)
    ax_base.legend(fontsize=12, frameon=False, loc="upper right")
    ax_base.tick_params(labelsize=12)

    for lab in ["low", "mid", "high"]:
        mask = clusters == lab
        ax_raster.scatter(spk_min[mask], [lab.capitalize()] * mask.sum(), s=2, color=CLUST_COLORS[lab])
    ax_raster.set_ylabel("cluster\n(raster)", fontsize=13)
    ax_raster.axvline(split_time, color="crimson", ls="--", lw=1)
    ax_raster.tick_params(labelsize=12)

    if t_wave_ms is not None:
        for lab in ["low", "mid", "high"]:
            if lab not in wf_feat:
                continue
            m, s, n = np.asarray(wf_feat[lab]["mean"]), np.asarray(wf_feat[lab]["std"]), wf_feat[lab]["n"]
            ax_wave.plot(t_wave_ms, m, color=CLUST_COLORS[lab], lw=1.3, label=f"{lab.capitalize()} (n={n})")
            ax_wave.fill_between(t_wave_ms, m - s, m + s, color=CLUST_COLORS[lab], alpha=0.15, lw=0)
    ax_wave.set_xlim(-5, 5)  # matches the Spike class's own +/-5ms fit window elsewhere in the pipeline
    ax_wave.set_xlabel("Time from peak (ms)", fontsize=13)
    ax_wave.set_ylabel("avg waveform\n(a.u.)", fontsize=13)
    ax_wave.legend(fontsize=12, frameon=False)
    ax_wave.tick_params(labelsize=12)

    for x, props in zip([0, 1], [before, after]):
        bottom = 0
        for lab in ["low", "mid", "high"]:
            frac = props.get(lab, 0)
            ax_mix.bar(x, frac, bottom=bottom, color=CLUST_COLORS[lab], width=0.35)
            if frac > 0.06:
                ax_mix.text(x, bottom + frac / 2, f"{frac:.0%}", ha="center", va="center",
                            fontsize=11, color="white")
            bottom += frac
    ax_mix.set_xlim(-0.5, 1.5)
    ax_mix.set_xticks([0, 1], ["before", "after"])
    ax_mix.set_ylim(0, 1)
    ax_mix.set_title("cluster mix", fontsize=14)
    ax_mix.set_yticks([])
    ax_mix.tick_params(labelsize=12)

    for a in [ax_base, ax_raster, ax_wave]:
        sns.despine(ax=a)
    sns.despine(ax=ax_mix, left=True)

    return fig


def plot_shift_summary_funnel(cell_nums, baseline_changepoints, feature_overlap_df):
    """Funnel of baseline shift -> waveform shift -> still variable, plus which features carry it.

    Three narrow shared-y-axis bars (all cells -> baseline-shift cells -> cells with a waveform
    shift) followed by a wider per-feature breakdown of how often each one overlaps a baseline
    shift. Returns (fig, fig_legend, stats): the data figure, a completely separate standalone
    figure holding only the funnel legend (its own Figure object, not a subplot of `fig`), and
    stats with the counts printed alongside the figures.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_total = len(cell_nums)
    n_baseline_shift = sum(1 for cn in cell_nums if len(baseline_changepoints.get(f"c{cn}", [])) > 0)
    pct_baseline_shift = n_baseline_shift / n_total * 100

    cells_with_baseline_cp = {cid for cid, cps in baseline_changepoints.items() if len(cps) > 0}
    cells_checkable = cells_with_baseline_cp & set(feature_overlap_df["cell_id"].unique())
    cells_with_wf_shift = cells_checkable & set(
        feature_overlap_df.loc[feature_overlap_df["overlaps_baseline"], "cell_id"])

    n_checkable, n_shift = len(cells_checkable), len(cells_with_wf_shift)
    pct_shift = n_shift / n_checkable * 100

    still_mixed_by_cell = (feature_overlap_df[feature_overlap_df["cell_id"].isin(cells_with_wf_shift) &
                                               feature_overlap_df["overlaps_baseline"]]
                            .groupby("cell_id")["still_mixed"].any())
    n_mixed = int(still_mixed_by_cell.sum())
    n_with_shift = len(still_mixed_by_cell)
    pct_mixed = n_mixed / n_with_shift * 100

    feat_summary = feature_overlap_df.groupby("feature").agg(
        n_cells=("cell_id", "count"),
        n_overlapping=("overlaps_baseline", "sum"),
    ).reset_index()
    feat_summary["pct_overlapping"] = (feat_summary["n_overlapping"] / feat_summary["n_cells"] * 100).round(1)
    feat_summary = feat_summary.sort_values("pct_overlapping", ascending=False)
    feat_summary["feature_label"] = feat_summary["feature"].apply(strip_cluster)

    fig = plt.figure(figsize=(18, 5.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[3, 2.2], wspace=0.45)
    left = outer[0].subgridspec(1, 3, wspace=0.2)
    ax0 = fig.add_subplot(left[0])
    ax1 = fig.add_subplot(left[1], sharey=ax0)
    ax2 = fig.add_subplot(left[2], sharey=ax0)
    ax3 = fig.add_subplot(outer[1])

    panels = [
        (ax0, pct_baseline_shift, n_baseline_shift, n_total - n_baseline_shift, "baseline shift",
         "no baseline shift", "#444444", "all cells"),
        (ax1, pct_shift, n_shift, n_checkable - n_shift, "waveform shift", "no waveform shift",
         "#0072B2", "baseline-shift\ncells"),
        (ax2, pct_mixed, n_mixed, n_with_shift - n_mixed, "still variable", "clean switch",
         "#D55E00", "cells with a\nwaveform shift"),
    ]
    for i, (ax, pct, n_top, n_bot, top_label, bot_label, top_color, xlabel) in enumerate(panels):
        ax.bar(0, pct, color=top_color, width=0.5, label=top_label)
        ax.bar(0, 100 - pct, bottom=pct, color="lightgray", width=0.5, label=bot_label)
        ax.text(0, pct / 2, f"{n_top}/{n_top + n_bot}", ha="center", va="center", color="white", fontsize=21)
        ax.text(0, pct + (100 - pct) / 2, f"{n_bot}/{n_top + n_bot}", ha="center", va="center", fontsize=21)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([0], [xlabel], fontsize=18)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=16)
        sns.despine(ax=ax)
        if i > 0:
            ax.tick_params(labelleft=False)
            sns.despine(ax=ax, left=True)
    ax0.set_ylabel("% of cells", fontsize=19)

    # legend is a completely separate Figure, not a subplot of `fig`, so it can never overlap
    # or get clipped by the bar panels above.
    handles = [h for a in (ax0, ax1, ax2) for h in a.get_legend_handles_labels()[0]]
    labels = [l for a in (ax0, ax1, ax2) for l in a.get_legend_handles_labels()[1]]
    fig_legend = plt.figure(figsize=(14, 2))
    ax_legend = fig_legend.add_axes([0, 0, 1, 1])
    ax_legend.axis("off")
    fig_legend.legend(handles, labels, ncol=3, loc="center", fontsize=26, frameon=False)

    ax3.barh(feat_summary["feature_label"], feat_summary["pct_overlapping"], color="#0072B2")
    for y, (n_over, n_cells) in enumerate(zip(feat_summary["n_overlapping"], feat_summary["n_cells"])):
        ax3.text(feat_summary["pct_overlapping"].iloc[y] + 2, y, f"{n_over}/{n_cells}", va="center", fontsize=17)
    ax3.set_xlabel("% of cells where this feature\nchanges after the baseline shift", fontsize=17)
    ax3.set_xlim(0, 118)
    ax3.tick_params(axis="x", labelsize=18)
    ax3.tick_params(axis="y", labelsize=17)
    ax3.invert_yaxis()
    sns.despine(ax=ax3)

    stats = {
        "n_total": n_total, "n_baseline_shift": n_baseline_shift, "pct_baseline_shift": pct_baseline_shift,
        "n_checkable": n_checkable, "n_shift": n_shift, "pct_shift": pct_shift,
        "n_with_shift": n_with_shift, "n_mixed": n_mixed, "pct_mixed": pct_mixed,
        "feat_summary": feat_summary.drop(columns="feature_label"),
    }
    return fig, fig_legend, stats
