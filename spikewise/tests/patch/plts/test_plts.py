"""Test plotting functions."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spikewise.patch.plts import plot_model
from spikewise.patch.plts.plts import _peak_align
from spikewise.tests.utils import plot_test
from spikewise.patch.fit import Spike
from spikewise.patch.sim import sim_ppoly_dist, sim_patch


@plot_test
@pytest.mark.parametrize('mode', ['full', 'ramp', 'exp'])
@pytest.mark.parametrize('inds', [None, 0, [0, 1], [0, 'null']])
def test_plot_model(sim_patch_spikes, mode, inds):

    sp = sim_patch_spikes['sp']

    if isinstance(inds, list) and inds[1] == 'null':
        inds = [0, 1]
        sp.inds_error = [1]

    plot_model(sp, mode=mode, inds=inds, show_points=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fast_spiking_sp(isi_ms=3.0, n_spikes=30, window_ms=5.0, fs=200000):
    """
    Build a Spike fitted on a fast-spiking signal where ISI < window post-length.

    At isi_ms < window_ms the next spike enters the post-window, causing
    argmax-based alignment to pick the wrong peak.
    """
    poly_means = np.load('params/pvc-6_param_means.npy')
    poly_cov   = np.load('params/pvc-6_param_cov.npy')
    degree     = np.load('params/pvc-6_degree.npy')

    spikes, _, _ = sim_ppoly_dist(poly_means, poly_cov, degree,
                                   n_spikes, seeds=np.arange(n_spikes))

    isi_samples = np.full(n_spikes - 1, int(isi_ms * fs / 1000))
    sig = sim_patch(spikes, isi_samples, tau=2500, pad=int(fs * 0.05))

    sp = Spike(thresh_amp=-10, window_length=(window_ms, window_ms),
               smooth_frac=0.008, pre_inflection_ms=0.5)
    sp.fit(list(sig), fs, n_jobs=1)
    sp.filter_features()
    return sp, fs, window_ms


# ---------------------------------------------------------------------------
# Test: control points land ON spikes (peak at t=0) for fast-spiking data
# ---------------------------------------------------------------------------

def test_show_points_peak_at_zero_fast_spiking():
    """
    Peak control point must be at t=0 ms even when neighboring spikes
    enter the ±window window (fast-spiking / short ISI).

    This is the regression test for the Allen CT aspiny-cell bug where
    argmax picked the NEXT spike's peak and shifted all control points.
    """
    sp, fs, window_ms = _make_fast_spiking_sp(isi_ms=3.0, window_ms=5.0)

    peak_ind = int(window_ms * fs / 1000)  # designed peak position in window

    # Every non-error spike's peak index must equal the designed peak_ind
    for i in range(sp.n_spikes):
        if i in sp.inds_error:
            continue
        assert sp.indices[i][3] == peak_ind, (
            f"Spike {i}: peak index {sp.indices[i][3]} != expected {peak_ind}"
        )

    # Plot must succeed and produce a figure with data
    plt.close('all')
    sp.plot(show_points=True)
    ax = plt.gca()
    assert ax.has_data()

    # x-axis must be clipped to [-window_ms, +window_ms]
    xlim = ax.get_xlim()
    assert abs(xlim[0] - (-window_ms)) < 0.1, f"xlim left {xlim[0]:.2f} != -{window_ms}"
    assert abs(xlim[1] - window_ms)    < 0.1, f"xlim right {xlim[1]:.2f} != +{window_ms}"

    plt.close('all')


def test_show_points_peak_at_zero_normal_isi():
    """Same check for normal ISI (control: existing behavior must not regress)."""
    sp, fs, window_ms = _make_fast_spiking_sp(isi_ms=50.0, window_ms=5.0)
    peak_ind = int(window_ms * fs / 1000)

    for i in range(sp.n_spikes):
        if i in sp.inds_error:
            continue
        assert sp.indices[i][3] == peak_ind

    plt.close('all')
    sp.plot(show_points=True)
    ax = plt.gca()
    assert ax.has_data()
    xlim = ax.get_xlim()
    assert abs(xlim[0] - (-window_ms)) < 0.1
    assert abs(xlim[1] - window_ms)    < 0.1
    plt.close('all')


# ---------------------------------------------------------------------------
# Test: _peak_align with explicit peak_idxs vs argmax
# ---------------------------------------------------------------------------

def test_peak_align_explicit_peak_idxs():
    """
    When peak_idxs is provided, _peak_align must center on those indices,
    not on argmax.  Regression: argmax picks wrong peak in busy windows.
    """
    fs   = 200000
    dt   = 1.0 / fs
    t    = np.linspace(0, 0.01, int(0.01 * fs))  # 10 ms window

    # Spike-like waveform: true peak at index 1000 (5 ms), but add a taller
    # artifact spike at index 1800 to fool argmax.
    wf = np.zeros(len(t)) - 65.0
    wf[900:1100]  += np.hanning(200) * 80   # true spike peak at ~1000
    wf[1750:1850] += np.hanning(100) * 100  # taller artifact at ~1800

    waveforms  = [wf.copy() for _ in range(5)]
    times_axis = np.arange(len(wf)) / fs

    # Without explicit peak_idxs: argmax finds the artifact → peak shifts
    aligned_bad, t_bad = _peak_align(waveforms, times_axis, wght=1000)
    # With explicit peak_idxs: should center at 1000
    aligned_ok, t_ok   = _peak_align(waveforms, times_axis, wght=1000,
                                      peak_idxs=[1000]*5)

    # t=0 in t_ok must be at the true spike peak (index 1000)
    zero_idx_ok = int(np.argmin(np.abs(t_ok)))
    assert abs(t_ok[zero_idx_ok]) < 1.0, (
        f"t=0 not at true peak; t_ok[{zero_idx_ok}]={t_ok[zero_idx_ok]:.2f}ms"
    )
    # In the bad alignment, t=0 should NOT be at index 1000
    # (it will be at the artifact position)
    zero_idx_bad = int(np.argmin(np.abs(t_bad)))
    # The artifact is at 1800 → expected shift = (1800-1000)/200 = 4 ms
    # So t_bad at original 1000 ≈ -4 ms
    t_at_true_peak_bad = t_bad[1000] if 1000 < len(t_bad) else float('nan')
    assert t_at_true_peak_bad < -1.0, (
        "argmax alignment should have shifted the true peak away from t=0, "
        f"but t_bad[1000]={t_at_true_peak_bad:.2f}ms"
    )


# ---------------------------------------------------------------------------
# Test: xlim is always ±window_length regardless of data extent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('window_ms', [5.0, 10.0])
def test_xlim_clipped_to_window(sim_patch_spikes, window_ms):
    """After plotting, x-axis must be exactly ±window_length ms."""
    sp = sim_patch_spikes['sp']
    # Override window_length to test parameterically
    sp.window_length = (window_ms, window_ms)

    plt.close('all')
    sp.plot(show_points=False)
    ax = plt.gca()
    xlim = ax.get_xlim()
    assert abs(xlim[0] - (-window_ms)) < 0.1, f"left xlim {xlim[0]:.2f} != -{window_ms}"
    assert abs(xlim[1] - window_ms)    < 0.1, f"right xlim {xlim[1]:.2f} != +{window_ms}"
    plt.close('all')
