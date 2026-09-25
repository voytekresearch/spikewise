"""Helpers for 2_electrical_stimulation_causally_influences_spike_waveform.ipynb."""
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

sys.path.append(str(Path(__file__).resolve().parents[1] / 'AP_empirical_paper_all_analyses' / 'datasets' / 'pvc-6' / 'pvc6_helper_modules'))
from pvc6_plotting import (plot_confusion_matrix, plot_feature_importance_categorical,
                           plot_top_correlations_by_window, plot_beta_weights_combined, plot_window_expansion)
from widget_utils import show_figures

STIM_TYPES = ['constant', 'ramp', 'pink']
STIM_COLORS = {'constant': '#2E7D32', 'ramp': '#9933CC', 'pink': '#FF44CC'}
STIM_LABELS = {'constant': 'Constant current', 'ramp': 'Ramp current', 'pink': 'Pink noise'}
WINDOWS_MS = [5, 25, 50, 100, 200, 300, 400, 500]
TARGETS = ['stim_mean', 'stim_std', 'stim_exp']
TARGET_LABELS = {'stim_mean': 'Mean amplitude (pA)', 'stim_std': 'Standard deviation (pA)',
                 'stim_exp': 'Spectral exponent'}


# ── Dataset ─────────────────────────────────────────────────────────────────────

def plot_example_sweeps(data_dir):
    """Injected current (top) and membrane voltage (bottom) for one sweep of each stimulation type."""
    d = np.load(Path(data_dir) / 'example_sweeps.npz')
    fs = float(d['fs'])
    fig, axes = plt.subplots(2, 3, figsize=(16, 5.5), sharex='col',
                             gridspec_kw={'height_ratios': [1, 2]}, constrained_layout=True)
    for col, st in enumerate(STIM_TYPES):
        stim, volt = d[f'{st}_stim'], d[f'{st}_volt']
        t = np.arange(len(stim)) / fs
        axes[0, col].plot(t, stim, color=STIM_COLORS[st], lw=1)
        axes[1, col].plot(t, volt, color='k', lw=0.6)
        axes[0, col].set_title(f'{STIM_LABELS[st]} (sweep {int(d[f"{st}_sweep"])})', fontsize=13)
        axes[1, col].set_xlabel('Time (s)')
        for ax in axes[:, col]:
            ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('Current (pA)')
    axes[1, 0].set_ylabel('Voltage (mV)')
    plt.show()


# ── Stimulation type classifier ─────────────────────────────────────────────────

def plot_classifier_results(data_dir):
    """Out-of-fold accuracy across repeats, mean confusion matrix (Fig 3B) and feature importances (Fig 3C)."""
    d = np.load(Path(data_dir) / 'stim_type_classifier.npz')
    acc = d['accuracies']
    print(f'Out-of-fold accuracy: {acc.mean():.1%} +/- {acc.std():.1%} '
          f'across {len(acc)} repeated 5-fold splits (chance = 33.3%)')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(acc, bins=30, color='#00838F', edgecolor='white')
    ax.axvline(acc.mean(), color='black', lw=2, ls='--', label=f'mean = {acc.mean():.3f}')
    ax.axvline(1 / 3, color='#d62728', lw=2, ls=':', label='chance (0.333)')
    ax.set_xlabel('Out-of-fold accuracy')
    ax.set_ylabel('Repeats')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

    # the paper's plotting functions only need the class labels and feature names from the model/X
    model = SimpleNamespace(classes_=d['classes'])
    X = pd.DataFrame(columns=d['feature_names'])
    plot_confusion_matrix(model, X, None, cm=np.round(d['mean_confusion']).astype(int))
    plot_feature_importance_categorical(model, X, d['importances'])


# ── Pink noise ──────────────────────────────────────────────────────────────────

def load_ridge_results(data_dir):
    with gzip.open(Path(data_dir) / 'ridge_results_by_window.json.gz', 'rt') as fh:
        raw = json.load(fh)
    return {key: {f: (v if f == 'feature_names' else np.asarray(v)) for f, v in res.items()}
            for key, res in raw.items()}


def plot_pink_window_schematic(data_dir, window_ms=200):
    """One pink-noise spike: the pre-spike window used for the stimulus features, and their values."""
    d = np.load(Path(data_dir) / 'pink_example_spike.npz')
    fs, infl, peak = float(d['fs']), int(d['inflection_idx']), int(d['peak_idx'])
    stim, volt = d['stim'], d['volt']
    t = (np.arange(len(stim)) - peak) / fs * 1000
    start = infl - int(window_ms * fs / 1000)
    window = stim[start:infl]

    fig, (ax_stim, ax_volt) = plt.subplots(2, 1, figsize=(16, 5), sharex=True, constrained_layout=True)
    for ax in (ax_stim, ax_volt):
        ax.axvspan(t[start], t[infl], color=STIM_COLORS['pink'], alpha=0.15, lw=0)
        ax.axvline(t[infl], color='k', ls='--', lw=1)
        ax.spines[['top', 'right']].set_visible(False)
    ax_stim.plot(t, stim, color=STIM_COLORS['pink'], lw=0.8)
    ax_stim.set_ylabel('Current (pA)')
    ax_stim.set_title(f'{window_ms} ms window before the spike inflection point:  '
                      f'mean = {np.mean(window):.1f} pA,  SD = {np.std(window):.1f} pA', fontsize=13)
    ax_volt.plot(t, volt, color='k', lw=1)
    ax_volt.set_ylabel('Voltage (mV)')
    ax_volt.set_xlabel('Time from spike peak (ms)')
    plt.show()


def plot_pink_feature_distributions(pink_spikes, window_ms):
    """Distribution of each pre-spike stimulus feature across all pink-noise spikes, for one window."""
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(16, 3.8), constrained_layout=True)
    for ax, tgt in zip(axes, TARGETS):
        ax.hist(pink_spikes[f'{tgt}_{window_ms}ms'].dropna(), bins=30, color=STIM_COLORS['pink'], edgecolor='white')
        ax.set_xlabel(TARGET_LABELS[tgt])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Spikes')
    fig.suptitle(f'Pre-spike stimulus features across all pink-noise spikes ({window_ms} ms window)', fontsize=13)
    plt.show()


def plot_r2_across_windows(results):
    """Ridge R² for each stimulus target across all window sizes, against a shuffled control."""
    shuffled = {f'{w}ms_{t}': results[f'shuf_{w}ms_{t}'] for w in WINDOWS_MS for t in TARGETS}
    plot_window_expansion(results, WINDOWS_MS, targets=tuple(TARGETS),
                          target_labels=('stim mean', 'stim std', 'stim exp'), shuffle_results=shuffled)


def _window_slider(window_ms):
    return widgets.SelectionSlider(options=WINDOWS_MS, value=window_ms, description='Window (ms)',
                                   continuous_update=False, style={'description_width': 'initial'},
                                   layout=widgets.Layout(width='500px'))


def pink_window_schematic_explorer(data_dir, window_ms=200):
    """Slider over pre-spike window sizes; redraws the window schematic on the example spike."""
    slider = _window_slider(window_ms)
    plot_area = widgets.Output()

    def redraw(*_):
        with plot_area:
            plot_area.clear_output()
            show_figures(plot_pink_window_schematic, data_dir, window_ms=slider.value)

    slider.observe(redraw, names='value')
    display(slider, plot_area)
    redraw()


def pink_window_explorer(pink_spikes, results, window_ms=200):
    """Slider over pre-spike window sizes; redraws the stimulus feature distributions, top
    waveform/stimulus correlations (Fig 3E) and ridge regression results (Fig 3F-G) for that window."""
    slider = _window_slider(window_ms)
    plot_area = widgets.Output()

    def redraw(*_):
        w = slider.value
        df_w = pink_spikes.rename(columns={f'{t}_{w}ms': t for t in TARGETS}).dropna(subset=['stim_mean'])
        keys = [f'{w}ms_{t}' for t in TARGETS]
        with plot_area:
            plot_area.clear_output()
            show_figures(plot_pink_feature_distributions, pink_spikes, w)
            print(f'Ridge regression, {w} ms window: how well spike waveform features predict each stimulus feature')
            for t, k in zip(TARGETS, keys):
                r = results[k]
                print(f"  {TARGET_LABELS[t]:<24} R² = {float(r['r2_mean']):.2f}  "
                      f"95% CI [{r['r2_ci'][0]:.2f}, {r['r2_ci'][1]:.2f}]  permutation p = {float(r['p_val_perm']):.3f}")
            show_figures(plot_top_correlations_by_window, df_w, window_ms=w, top=True, quiet=True)
            show_figures(plot_beta_weights_combined, results, window_ms=w, quiet=True)

    slider.observe(redraw, names='value')
    display(slider, plot_area)
    redraw()
