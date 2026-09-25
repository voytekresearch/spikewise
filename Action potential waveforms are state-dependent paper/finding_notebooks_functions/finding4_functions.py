"""Helpers for 4_within_neuron_variability_exceeds_between_neuron_differences.ipynb."""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / 'AP_empirical_paper_all_analyses' / 'datasets' / 'spe-1' / 'spe1_helper_modules'))
from spk_feat_cluster_comp_analysis import plot_spike_to_avg_distances, plot_waveform_dist_sorted_dots

CELL_TYPES = {'PC': 'putative PC', 'IN': 'putative IN'}
C_MEAN, C_OTHER, C_DIFF = '#1B7F4C', '#555555', '#F2D48F'   # as in the Fig 4D nRMSE schematic


# ── Data ────────────────────────────────────────────────────────────────────────

def load_distance_data(data_dir):
    """Everything the notebook needs, from the files in finding_notebooks_data/finding4/."""
    data_dir = Path(data_dir)
    dots = np.load(data_dir / '_sorted_dots_v1.npz')
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
