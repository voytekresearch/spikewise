"""Helpers for 4_within_neuron_variability_exceeds_between_neuron_differences.ipynb."""
import sys
import warnings
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

sys.path.append(str(Path(__file__).resolve().parents[1] / 'AP_empirical_paper_all_analyses' / 'datasets' / 'spe-1' / 'spe1_helper_modules'))
from spk_feat_cluster_comp_analysis import plot_spike_to_avg_distances, plot_waveform_dist_sorted_dots
from widget_utils import show_figures

CELL_TYPES = {'PC': 'putative PC', 'IN': 'putative IN'}
C_MEAN, C_OTHER, C_DIFF = '#1B7F4C', '#555555', '#F2D48F'   # as in the Fig 4D nRMSE schematic


# ── Data ────────────────────────────────────────────────────────────────────────

def load_distance_data(data_dir):
    """Everything the notebook needs, from the files in finding_notebooks_data/finding4/."""
    data_dir = Path(data_dir)
    dots = np.load(data_dir / '_sorted_dots_v2.npz')
    cell_ids = [str(c) for c in dots['cell_ids']]
    wf = np.load(data_dir / 'cell_waveforms.npz')
    return {
        'dir': data_dir,
        'cells': pd.read_csv(data_dir / 'cells.csv').set_index('cell_id'),
        'within': {c: dots[f'nrmse_{i}'] for i, c in enumerate(cell_ids)},
        'between': dots['btw_nrmse_all'],
        'mean': {c: wf[f'{c}__mean'].astype(float) for c in cell_ids},
        'spikes': {c: wf[f'{c}__spikes'].astype(float) for c in cell_ids},
        'fs': {c: float(wf[f'{c}__fs']) for c in cell_ids},
    }


def _cell_label(data, cid):
    ct = data['cells'].loc[cid, 'cell_type']
    return f'{cid} ({CELL_TYPES.get(ct, ct)}, {len(data["within"][cid])} spikes)'


# ── Population results and paper panels ─────────────────────────────────────────

def print_within_between_summary(data):
    v2 = np.load(data['dir'] / '_spike_to_avg_distances_v2.npz')
    print(f"nRMSE, median:              within cell {np.median(v2['within_nrmse']):.3f}   "
          f"between cells {np.median(v2['Between_(all)_nrmse']):.3f}")
    print(f"cosine similarity, median:  within cell {np.median(v2['within_cos']):.3f}   "
          f"between cells {np.median(v2['Between_(all)_cos']):.3f}")

    btw = np.median(data['between'])
    pct = pd.Series({c: np.mean(v > btw) * 100 for c, v in data['within'].items()}).sort_values(ascending=False)
    above = [c for c, v in data['within'].items() if np.median(v) > btw]
    print(f'\nCells whose median single spike nRMSE exceeds the between-cell median: '
          + ', '.join(f'{c} ({pct[c]:.1f}% of spikes above)' for c in pct.index if c in above))
    print(f'Cells with more than 25% of spikes above the between-cell median: {(pct > 25).sum()}')
    print(f'Cells with more than 10% of spikes above the between-cell median: {(pct > 10).sum()}')


def plot_fig4d(data):
    """Single spike-to-own-average nRMSE for every cell, sorted by within-cell variability (Fig 4D)."""
    # same call as pop_within_vs_between_waveform.ipynb, reading the shipped copy of its cache
    plot_waveform_dist_sorted_dots(data['cells'].reset_index(), wf_dir=data['dir'], spike_fit_dir=None,
                                   cache_dir=data['dir'])


def plot_within_vs_between_groups(data):
    """Within-cell nRMSE against between-cell nRMSE, overall and within cell type / recording method (Supp Fig 14A-B)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plot_spike_to_avg_distances(data['cells'].reset_index(), wf_dir=data['dir'], spike_fit_dir=None,
                                    cache_dir=data['dir'], metrics=['nRMSE'])


# ── Explorer ────────────────────────────────────────────────────────────────────

def _nrmse(spikes, mean, denom):
    return np.sqrt(np.mean((spikes - mean) ** 2, axis=-1)) / denom


def plot_cell_comparison(data, cell, other):
    """One cell's single spikes against its own average, against another cell's average, and the distributions."""
    mean_a, mean_b, spikes = data['mean'][cell], data['mean'][other], data['spikes'][cell]
    peak_a = np.max(np.abs(mean_a))
    denom_ab = max(peak_a, np.max(np.abs(mean_b)))
    within_sample = _nrmse(spikes, mean_a, peak_a)
    to_other = _nrmse(spikes, mean_b, denom_ab)
    within_all = data['within'][cell]
    btw = data['between']
    btw_med, btw_q25, btw_q75 = np.median(btw), *np.percentile(btw, [25, 75])
    t = (np.arange(len(mean_a)) - len(mean_a) // 2) / data['fs'][cell] * 1000

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1, 1, 1.3]})

    # a) single spikes vs the cell's own average, highlighting a typical one (median nRMSE)
    ax = axes[0]
    typ = int(np.argsort(within_sample)[len(within_sample) // 2])
    for s in spikes[np.argsort(within_sample)[::max(1, len(spikes) // 25)]]:
        ax.plot(t, s, color='#bbbbbb', lw=0.6, alpha=0.6)
    ax.fill_between(t, mean_a, spikes[typ], color=C_DIFF, alpha=0.8, lw=0)
    ax.plot(t, mean_a, color=C_MEAN, lw=3, label=f'{cell} average')
    ax.plot(t, spikes[typ], color='k', lw=1.8, label=f'typical single spike (nRMSE {within_sample[typ]:.2f})')
    ax.set_title(f'{cell}: single spikes vs its own average', fontsize=13)
    ax.set_ylabel('Voltage')
    ax.legend(frameon=False, fontsize=9, loc='upper right')

    # b) the two cells' averages
    ax = axes[1]
    between_nrmse = np.sqrt(np.mean((mean_a - mean_b) ** 2)) / denom_ab
    ax.fill_between(t, mean_a, mean_b, color=C_DIFF, alpha=0.8, lw=0)
    ax.plot(t, mean_a, color=C_MEAN, lw=3, label=f'{cell} average')
    ax.plot(t, mean_b, color=C_OTHER, lw=3, label=f'{other} average')
    ax.set_title(f'{cell} vs {other} averages (nRMSE {between_nrmse:.2f})', fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    for ax in axes[:2]:
        ax.set_xlabel('Time from peak (ms)')
        ax.spines[['top', 'right']].set_visible(False)

    # c) distributions
    ax = axes[2]
    top = np.percentile(np.concatenate([within_all, to_other, btw]), 99.5)
    bins = np.linspace(0, top, 60)
    ax.axvspan(btw_q25, btw_q75, color='#CCCCCC', alpha=0.5, lw=0, label='between cells, all pairs (IQR)')
    ax.axvline(btw_med, color='#555555', ls='--', lw=2, label='between cells, all pairs (median)')
    ax.hist(within_all, bins=bins, density=True, color=C_MEAN, alpha=0.6,
            label=f'{cell} spikes vs {cell} average (all {len(within_all)})')
    ax.hist(to_other, bins=bins, density=True, histtype='step', color=C_OTHER, lw=2.5,
            label=f'{cell} spikes vs {other} average ({len(to_other)} sampled)')
    ax.set_xlabel('nRMSE')
    ax.set_ylabel('Density')
    ax.set_title(f'{np.mean(within_all > btw_med):.1%} of {cell} spikes differ from their own average '
                 'more than the typical pair of cells', fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.show()


def cell_comparison_explorer(data, cell='c27', other='c4'):
    """Menus to pick a cell and a second cell to compare it with."""
    options = [(_cell_label(data, c), c) for c in sorted(data['within'], key=lambda c: int(c[1:]))]
    cell_menu = widgets.Dropdown(options=options, value=cell, description='Cell:',
                                 layout=widgets.Layout(width='360px'))
    other_menu = widgets.Dropdown(options=options, value=other, description='Compare with:',
                                  style={'description_width': 'initial'}, layout=widgets.Layout(width='400px'))
    plot_area = widgets.Output()

    def redraw(*_):
        with plot_area:
            plot_area.clear_output()
            show_figures(plot_cell_comparison, data, cell_menu.value, other_menu.value)

    cell_menu.observe(redraw, names='value')
    other_menu.observe(redraw, names='value')
    display(widgets.HBox([cell_menu, other_menu]), plot_area)
    redraw()
