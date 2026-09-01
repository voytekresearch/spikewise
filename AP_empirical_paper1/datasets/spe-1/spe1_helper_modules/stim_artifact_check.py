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
import numpy as np
from scipy.signal import welch, find_peaks, peak_widths

try:
    from .config import DICT_PATCH_FS, DICT_SPK_THRESH, DICT_PATCH_TYPE, SPE1_DATA_ROOT
    from .spk_feat_cluster_analysis import load_or_compute
except ImportError:
    from config import DICT_PATCH_FS, DICT_SPK_THRESH, DICT_PATCH_TYPE, SPE1_DATA_ROOT
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
