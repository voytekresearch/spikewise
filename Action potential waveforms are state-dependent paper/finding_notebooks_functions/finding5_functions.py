"""Helpers for 5_ap_waveform_predicts_peri_spike_lfp_state.ipynb."""
import contextlib
import gzip
import io
import json
import sys
import warnings
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

sys.path.append(str(Path(__file__).resolve().parents[1] / 'AP_empirical_paper_all_analyses' / 'datasets' / 'spe-1' / 'spe1_helper_modules'))
from pop_ridge_utils import (run_fraction_sig_tests, run_r2_tests, run_population_tests,
                             plot_r2_summary_boxplot, plot_beta_significant_summary, plot_r2_summary)
from widget_utils import show_figures

IAP_COLOR, EAP_COLOR, LFP_COLOR = '#1A8A8A', '#8B1A1A', '#8B1A1A'
CORE_TARGETS = ['pre_lfp_amp', 'pre_lfp_std', 'post_lfp_amp', 'post_lfp_std',
                'pre_gamma_auc', 'pre_exponent', 'pre_theta_auc',
                'post_gamma_auc', 'post_exponent', 'post_theta_auc']


# ── Dataset ─────────────────────────────────────────────────────────────────────

def plot_example_recording(data_dir):
    """Two seconds of the simultaneous recordings, plus the cell's average intracellular and extracellular spike."""
    d = np.load(Path(data_dir) / 'example_recording.npz')
    a = np.load(Path(data_dir) / 'average_spikes.npz')
    t0 = float(d['t0'])
    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[3.2, 1])
    rows = [('patch', 'Patch (IAP)', IAP_COLOR, 0.6), ('npx', 'Neuropixels (EAP)', EAP_COLOR, 0.5),
            ('lfp', 'Neuropixels LFP', LFP_COLOR, 1.2)]
    for r, (key, label, color, lw) in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0])
        x = d[key]
        t = t0 + np.arange(len(x)) / float(d[f'{key}_fs'])
        ax.plot(t, x, color=color, lw=lw)
        if key != 'lfp':
            for ts in d['spike_times_s']:
                ax.axvline(ts, color='0.75', lw=0.8, zorder=0)
        ax.set_ylabel(label)
        ax.set_xlim(t[0], t[-1])
        ax.spines[['top', 'right']].set_visible(False)
        if r < 2:
            ax.set_xticklabels([])
    ax.set_xlabel('Time in recording (s)')
    fig.axes[0].set_title(f'Cell {int(d["cell"])}: simultaneous recordings (grey lines = patch spikes)', fontsize=13)

    for r, (key, label, color) in enumerate([('iap', 'Average IAP', IAP_COLOR), ('eap', 'Average EAP', EAP_COLOR)]):
        ax = fig.add_subplot(gs[r, 1])
        m, sd, fs = a[f'{key}_mean'], a[f'{key}_sd'], float(a[f'{key}_fs'])
        t = (np.arange(len(m)) - len(m) // 2) / fs * 1000
        ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.2, lw=0)
        ax.plot(t, m, color=color, lw=2)
        ax.set_title(f'{label} (n = {int(a["n_spikes"])} spikes)', fontsize=12)
        ax.set_xlabel('Time from spike (ms)')
        ax.spines[['top', 'right']].set_visible(False)
    plt.show()


# ── Ridge regression ────────────────────────────────────────────────────────────

def load_ridge_population(data_dir):
    """Per-cell CV R², significance flags and betas for all 39 cells, as used by pop_ridge_utils."""
    with gzip.open(Path(data_dir) / 'ridge_population.json.gz', 'rt') as fh:
        raw = json.load(fh)
    arr = lambda d: {k: (arr(v) if isinstance(v, dict) else np.asarray(v, float)) for k, v in d.items()}
    raw['r2_pop'], raw['sig_pop'], raw['beta_pop'] = arr(raw['r2_pop']), arr(raw['sig_pop']), arr(raw['beta_pop'])
    raw['sig_pop'] = {t: {p: v.astype(bool) for p, v in d.items()} for t, d in raw['sig_pop'].items()}
    return raw


def run_population_stats(pop, alpha=0.05, min_cells=5):
    """Same population tests as pop_ridge_regression.ipynb (fraction significant, R² > 0, beta consistency)."""
    tn, tl, ps = pop['target_names'], pop['target_labels'], pop['predictor_sets']
    with contextlib.redirect_stdout(io.StringIO()):
        df_frac = run_fraction_sig_tests(pop['sig_pop'], tn, tl, ps, min_cells=min_cells, alpha=alpha)
        df_r2 = run_r2_tests(pop['r2_pop'], tn, tl, ps, min_cells=min_cells, alpha=alpha)
        sig_r2_targets = df_r2[df_r2['sig_r2'] & (df_r2['predictor_set'] == 'Waveform only')]['target'].tolist()
        df_tests = run_population_tests(pop['beta_pop'], tn, tl, min_cells=min_cells, alpha=alpha,
                                        sig_r2_targets=sig_r2_targets)
    return df_frac, df_r2, df_tests


def print_fraction_significant(pop, df_frac):
    rows = df_frac[df_frac['target'].isin(['pre_lfp_amp', 'post_lfp_amp', 'pre_lfp_std', 'post_lfp_std'])
                   & df_frac['predictor_set'].isin(['Waveform only', 'Log ISI only'])]
    print(f"Cells with a significant prediction (of {len(pop['cell_ids'])}):")
    for tgt, g in rows.groupby('target', sort=False):
        w = g[g['predictor_set'] == 'Waveform only']['frac_sig'].iloc[0]
        c = g[g['predictor_set'] == 'Log ISI only']['frac_sig'].iloc[0]
        print(f"  {g['target_label'].iloc[0]:<14} waveform {w:5.0%}   vs   log-ISI control {c:5.0%}")


def plot_r2_boxplot(pop, df_r2):
    """Fig 4G, left: per-cell CV R², waveform model vs log-ISI control."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # harmless matplotlib layout warnings from the paper's plotting code
        plot_r2_summary_boxplot(pop['r2_pop'], pop['sig_pop'], df_r2, pop['target_names'], pop['target_labels'],
                                predictor_set='Waveform only')


def plot_all_lfp_features(pop, df_r2):
    """Mean CV R² for every LFP feature (time domain and spectral), waveform model."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        core = [(t, l) for t, l in zip(pop['target_names'], pop['target_labels']) if not t.endswith('_hpf')]
        plot_r2_summary(pop['r2_pop'], df_r2, [t for t, _ in core], [l for _, l in core])


def plot_significant_betas(pop, df_tests):
    """Fig 4G, right: waveform features with a consistent direction across cells."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plot_beta_significant_summary(df_tests, pop['target_names'], pop['target_labels'])


def _plot_cell(pop, cell):
    i = pop['cell_ids'].index(cell)
    labels = {t: l for t, l in zip(pop['target_names'], pop['target_labels'])}
    targets = [t for t in CORE_TARGETS if t in pop['r2_pop']]
    x = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    for off, ps, color in [(-0.2, 'Waveform only', '#2E8B8B'), (0.2, 'Log ISI only', '0.6')]:
        r2 = np.array([pop['r2_pop'][t][ps][i] for t in targets])
        sig = np.array([pop['sig_pop'][t][ps][i] for t in targets])
        bars = ax.bar(x + off, r2, width=0.38, color=color, label=ps.replace('Log ISI only', 'Log-ISI control'))
        for b, s in zip(bars, sig):
            if s:
                ax.text(b.get_x() + b.get_width() / 2, max(b.get_height(), 0), '*', ha='center', va='bottom', fontsize=16)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[t].replace(' ', '\n', 1) for t in targets], fontsize=10)
    ax.set_ylabel('CV R²')
    ax.set_title(f"Cell {cell[1:]} ({pop['cell_types'][cell]}): how well spike shape predicts each LFP feature "
                 f"(* = significant for this cell)", fontsize=12)
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.show()


def cell_explorer(pop, cell='c3'):
    """Dropdown to pick a cell and see its CV R² for each LFP feature, waveform model vs log-ISI control."""
    options = [(f"Cell {c[1:]} ({pop['cell_types'][c]})", c) for c in sorted(pop['cell_ids'], key=lambda c: int(c[1:]))]
    menu = widgets.Dropdown(options=options, value=cell, description='Cell:')
    plot_area = widgets.Output()

    def redraw(*_):
        with plot_area:
            plot_area.clear_output()
            show_figures(_plot_cell, pop, menu.value)

    menu.observe(redraw, names='value')
    display(menu, plot_area)
    redraw()
