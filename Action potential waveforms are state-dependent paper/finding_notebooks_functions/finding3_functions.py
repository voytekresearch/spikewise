"""Helpers for 3_neurons_show_multimodal_spontaneous_variability.ipynb."""
import sys
import pickle
import tempfile
from itertools import combinations
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy.stats import zscore, spearmanr

sys.path.append(str(Path(__file__).resolve().parents[1] / 'AP_empirical_paper_all_analyses' / 'datasets' / 'spe-1' / 'spe1_helper_modules'))
import spk_feat_cluster_analysis
from spk_feat_cluster_analysis import (plot_clusters_over_time_min, cluster_transition_matrix,
                                       visualize_feature_groups_hist, compute_cluster_statistics)
from spk_feat_cluster_comp_analysis import (CLUST_COLORS, plot_cluster_legend,
                                            plot_feature_distribution_grid, plot_population_waveform_grid)
from widget_utils import show_figures

# the per-cell report helpers default to an older palette; use the paper's (blue = low, gold = mid, pink = high)
spk_feat_cluster_analysis.CLUSTER_COLORS.update(CLUST_COLORS)

SPK_FEATS = ['ramp_amp', 'inflection_time', 'inflection_amp',
             'peak_amp', 'peak_width', 'peak_sharpness',
             'exp_lambda', 'exp_const', 'log_isi']
FEAT_LABELS = {'ramp_amp': 'Ramp amplitude', 'inflection_time': 'Inflection time',
               'inflection_amp': 'Inflection amplitude', 'peak_amp': 'Peak amplitude',
               'peak_width': 'Peak width', 'peak_sharpness': 'Peak sharpness',
               'exp_lambda': 'Decay rate (λ)', 'exp_const': 'Decay constant', 'log_isi': 'log ISI'}
CELL_TYPES = {'PC': 'putative PC', 'IN': 'putative IN'}
ORDER = ['low', 'mid', 'high']


# ── Data ────────────────────────────────────────────────────────────────────────

def load_cluster_data(data_dir):
    """Everything the notebook needs, from the files in finding_notebooks_data/finding3/."""
    data_dir = Path(data_dir)
    feats = pd.read_csv(data_dir / 'cluster_features.csv.gz', low_memory=False)
    clust_cols = [c for c in feats.columns if c.endswith('_cluster')]
    feats[clust_cols] = feats[clust_cols].astype(object).where(feats[clust_cols].notna(), None)
    return {
        'features': feats,
        'metrics': pd.read_csv(data_dir / 'cluster_metrics.csv'),
        'cells': pd.read_csv(data_dir / 'cells.csv').set_index('cell'),
        'waveforms': dict(np.load(data_dir / 'cluster_waveforms.npz')),
        'examples': dict(np.load(data_dir / 'example_spikes.npz')),
    }


def _cell_df(data, cell):
    df = data['features'][data['features'].cell == cell]
    return df.dropna(axis=1, how='all').reset_index(drop=True)


def clustered_features(data, cell):
    return [c[:-8] for c in _cell_df(data, cell).columns if c.endswith('_cluster')]


def _cluster_waveforms(data, cell, feature):
    """{'t_axis': samples from peak, label: {'mean', 'std', 'n'}} for one cell and clustered feature."""
    prefix = f'c{cell}__{feature}_cluster__'
    wf = {'t_axis': data['waveforms'][prefix + 't']}
    for lab in ORDER:
        if prefix + f'{lab}__mean' in data['waveforms']:
            wf[lab] = {k: data['waveforms'][prefix + f'{lab}__{k}'] for k in ('mean', 'std', 'n')}
    return wf


_PICKLE_DIRS = {}


def _paper_pickle_dir(data):
    """The paper's population plotting functions read per-cell pickles from a folder; write them once
    from the shipped data into a temporary folder so those functions can run unchanged."""
    key = id(data)
    if key not in _PICKLE_DIRS:
        tmp = Path(tempfile.mkdtemp(prefix='finding3_'))
        for cell in data['cells'].index:
            _cell_df(data, cell).to_pickle(tmp / f'c{cell}_cluster_df.pkl')
            feats = clustered_features(data, cell)
            if feats:
                with open(tmp / f'c{cell}_cluster_waveforms.pkl', 'wb') as fh:
                    pickle.dump({f'{f}_cluster': _cluster_waveforms(data, cell, f) for f in feats}, fh)
        _PICKLE_DIRS[key] = tmp
    return _PICKLE_DIRS[key]


def _cluster_table(data):
    """One row per (cell, clustered feature) with its number of clusters, as the paper's df_master."""
    rows = [{'cell_id': f'c{cell}', 'spike_feature': f,
             'num_clusters': _cell_df(data, cell)[f'{f}_cluster'].nunique()}
            for cell in data['cells'].index for f in clustered_features(data, cell)]
    return pd.DataFrame(rows)


# ── Population summary and paper panels ─────────────────────────────────────────

def print_cluster_summary(data):
    table = _cluster_table(data)
    n_cells = len(data['cells'])
    wf = table[table.spike_feature != 'log_isi']
    print(f'{wf.cell_id.nunique()} of {n_cells} neurons ({wf.cell_id.nunique() / n_cells:.0%}) have multimodal '
          f'clusters in at least one waveform feature')
    for f, n in wf.groupby('spike_feature').cell_id.nunique().sort_values(ascending=False).items():
        print(f'  {FEAT_LABELS[f]:<22} {n:>2} cells ({n / n_cells:.1%})')
    n_isi = (table.spike_feature == 'log_isi').sum()
    print(f'{n_isi} of {n_cells} neurons ({n_isi / n_cells:.1%}) have multimodal ISI distributions')


def plot_fig4b(data):
    """Within-cell feature distributions split by cluster, for the example cells in Fig 4B."""
    # Same selection and call as pop_spk_waveform_clusters.ipynb ("Feature distribution grid, main figure")
    best_cells = {
        'peak_width':     ['c23', 'c25'],
        'peak_amp':       ['c4',  'c32'],
        'peak_sharpness': ['c21', 'c32'],
        'log_isi':        ['c1',  'c20'],
        'inflection_time': [],   # not shown in the paper figure
    }
    merged = [{'panels': [('c19', 'exp_lambda'), ('c27', 'inflection_amp')], 'is_isi': False}]
    plot_cluster_legend(fontsize=22)
    plt.show()
    plot_feature_distribution_grid(_cluster_table(data), _paper_pickle_dir(data),
                                   cells_to_plot=best_cells, merged_rows=merged)
    plt.show()


def plot_fig4c(data):
    """Cluster-average waveforms (mean +/- SD) for the example cells in Fig 4C."""
    # Same selection and call as pop_spk_waveform_clusters.ipynb ("Population waveform grid, main figure")
    best_wf_panels = [
        (4,  'peak_amp'),
        (3,  'peak_amp'),
        (19, 'exp_lambda'),
        (45, 'exp_lambda'),
        (21, 'peak_sharpness'),
        (32, 'peak_sharpness'),
        (42, 'peak_width'),
        (21, 'peak_amp'),
    ]
    plot_population_waveform_grid(_paper_pickle_dir(data), panels_to_plot=best_wf_panels, cols=2,
                                  title_fontsize=18, all_black=True, facecolor='white', show_legend=False)
    plt.show()


# ── Per-cell cluster report ─────────────────────────────────────────────────────

def _plot_cluster_spikes(data, cell, feature, groups):
    """Example single spikes (thin) and the cluster-average waveform (mean +/- SD over all spikes)."""
    df = _cell_df(data, cell)
    fs = data['cells'].loc[cell, 'fs']
    wf = _cluster_waveforms(data, cell, feature)
    ids, spikes = data['examples'][f'c{cell}__spk_id'], data['examples'][f'c{cell}__spikes'].astype(float)
    t_ex = (np.arange(spikes.shape[1]) - spikes.shape[1] // 2) / fs * 1000
    labels = df.set_index('spk_id')[f'{feature}_cluster']

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), sharey=True, constrained_layout=True)
    for ax, lab in zip(np.atleast_1d(axes), groups):
        color = CLUST_COLORS[lab]
        for k in np.flatnonzero(labels.reindex(ids).to_numpy() == lab):
            ax.plot(t_ex, spikes[k], color=color, lw=0.5, alpha=0.35)
        t = wf['t_axis'] / fs * 1000
        m, s = wf[lab]['mean'][:len(t)], wf[lab]['std'][:len(t)]
        ax.fill_between(t, m - s, m + s, color=color, alpha=0.25, lw=0)
        ax.plot(t, m, color='k', lw=2)
        ax.set_title(f'{lab} cluster (n = {(labels == lab).sum()} spikes)', color=color, fontsize=13)
        ax.set_xlim(-4, 4)
        ax.set_xlabel('Time from peak (ms)')
        ax.spines[['top', 'right']].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel('Voltage')
    fig.suptitle(f'c{cell}: spikes grouped by {FEAT_LABELS[feature]} cluster '
                 '(thin: example spikes; black and shading: cluster mean ± SD, outlier waveforms excluded)', fontsize=13)
    plt.show()


def _plot_pairwise_waveforms(data, cell, feature, groups):
    """Each pair of cluster averages in absolute amplitude, max-scaled (nRMSE) and z-scored (shape; cosine similarity)."""
    fs = data['cells'].loc[cell, 'fs']
    wf = _cluster_waveforms(data, cell, feature)
    t = wf['t_axis'] / fs * 1000
    met = data['metrics'][(data['metrics'].cell == cell) & (data['metrics'].feature_clustered == feature)]
    pairs = list(combinations(groups, 2))

    fig, axes = plt.subplots(len(pairs), 3, figsize=(16, 3.8 * len(pairs)), squeeze=False, constrained_layout=True)
    for row, (g1, g2) in zip(axes, pairs):
        w1, w2 = wf[g1]['mean'][:len(t)], wf[g2]['mean'][:len(t)]
        ok = np.isfinite(w1) & np.isfinite(w2)
        scale = max(np.abs(w1[ok]).max(), np.abs(w2[ok]).max())
        m = met[met.groups.isin([f'{g1.title()}-{g2.title()}', f'{g2.title()}-{g1.title()}'])]
        nrmse, cos = (m.nRMSE.iloc[0], m.cos_sim.iloc[0]) if len(m) else (np.nan, np.nan)
        z1, z2 = np.full_like(w1, np.nan), np.full_like(w2, np.nan)
        z1[ok], z2[ok] = zscore(w1[ok]), zscore(w2[ok])
        panels = [(w1, w2, 'Absolute amplitude', 'Voltage', None),
                  (w1 / scale, w2 / scale, 'Relative amplitude (max-scaled)', 'Norm. amplitude', f'nRMSE = {nrmse:.3f}'),
                  (z1, z2, 'Shape only (z-scored)', 'z-score', f'cosine similarity = {cos:.3f}')]
        for ax, (a, b, title, ylabel, stat) in zip(row, panels):
            ax.plot(t, a, color=CLUST_COLORS[g1], lw=2, label=g1)
            ax.plot(t, b, color=CLUST_COLORS[g2], lw=2, label=g2)
            ax.fill_between(t, a, b, color='gray', alpha=0.2, lw=0)
            ax.set_xlim(-4, 4)
            ax.set_title(f'{g1} vs {g2}: {title}', fontsize=12)
            ax.set_ylabel(ylabel)
            ax.spines[['top', 'right']].set_visible(False)
            if stat:
                ax.text(0.03, 0.95, stat, transform=ax.transAxes, va='top', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        row[0].legend(frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel('Time from peak (ms)')
    plt.show()


def _plot_transition_matrix(df, cluster_col, groups):
    trans = cluster_transition_matrix(df, cluster_col).loc[list(groups), list(groups)]
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(trans, annot=True, fmt='.2f', cmap='magma', cbar=False)
    plt.title('Probability of the next spike\'s cluster')
    plt.xlabel('Next spike cluster')
    plt.ylabel('Current spike cluster')
    plt.tight_layout()
    plt.show()


def _plot_cluster_stats(df, feature, groups):
    group_data = [df.loc[df[f'{feature}_cluster'] == g, feature].dropna().to_numpy() for g in groups]
    res = compute_cluster_statistics(group_data)
    if res.get('p_value') is None:
        print('  Not enough spikes to compare clusters.')
        return
    p_txt = f"p = {res['p_value']:.2g}" if res['p_value'] > 0 else 'p < 1e-300'
    print(f"  {res['test_name']}: {p_txt} {res['significance']}, "
          f"η² = {res['eta_squared']:.3f} ({res['interpretation']})")
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(group_data, labels=list(groups), patch_artist=True, showfliers=False)
    for patch, g in zip(bp['boxes'], groups):
        patch.set_facecolor(CLUST_COLORS[g])
    ax.set_ylabel(FEAT_LABELS[feature])
    ax.set_title(f'{FEAT_LABELS[feature]} by cluster')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_cluster_report(data, cell, feature):
    """The full per-cell cluster report (as in each spe-1_c{N}_clusters.ipynb) for one cell and clustered feature."""
    df = _cell_df(data, cell)
    col = f'{feature}_cluster'
    groups = [g for g in ORDER if g in set(df[col].dropna())]
    cell_type = CELL_TYPES[data['cells'].loc[cell, 'cell_type']]

    print(f'c{cell} ({cell_type}, {len(df)} spikes): {FEAT_LABELS[feature]} splits into {len(groups)} clusters '
          f'({", ".join(f"{g}: {(df[col] == g).sum()}" for g in groups)} spikes)\n')

    print('1. Spike waveforms by cluster')
    show_figures(_plot_cluster_spikes, data, cell, feature, groups)

    print('2. How different are the cluster-average waveforms?')
    show_figures(_plot_pairwise_waveforms, data, cell, feature, groups)

    print('3. When in the recording does each cluster occur? (top: every spike; bottom: firing rate per cluster)')
    ordinal = df[col].map({g: i for i, g in enumerate(groups)})
    rho, p = spearmanr(df['spk_times_ms'], ordinal, nan_policy='omit')
    print(f'   Spearman ρ between spike time and cluster label = {rho:.2f} ' + (f'(p = {p:.2g})' if p > 0 else '(p < 1e-300)'))
    show_figures(plot_clusters_over_time_min, df, time_col='spk_times_ms', label_col=col, time_unit='ms',
                 bin_size_ms=1000, sigma_bins=2)

    print("4. Does a spike's cluster predict the next spike's cluster?")
    show_figures(_plot_transition_matrix, df, col, groups)

    print('5. All spike features, split by these clusters')
    show_figures(visualize_feature_groups_hist, df, features=SPK_FEATS, group_col=col, groups=tuple(groups),
                 group_names=tuple(g.title() for g in groups), color_map=dict(CLUST_COLORS),
                 common_norm=True, alpha=0.5, quiet=True)

    print(f'6. Is {FEAT_LABELS[feature]} different between clusters?')
    show_figures(_plot_cluster_stats, df, feature, groups)


def cluster_report_explorer(data, cell=21, feature='peak_sharpness'):
    """Menus to pick a cell and one of its clustered features, and show the full cluster report."""
    cells = data['cells']
    options = [(f'c{c} ({CELL_TYPES[cells.loc[c, "cell_type"]]}, {cells.loc[c, "n_spikes"]} spikes)', c)
               for c in cells.index if clustered_features(data, c)]
    cell_menu = widgets.Dropdown(options=options, value=cell, description='Cell:',
                                 layout=widgets.Layout(width='340px'))
    feat_menu = widgets.Dropdown(description='Feature:', layout=widgets.Layout(width='300px'))
    plot_area = widgets.Output()

    def feature_options(c):
        return [(FEAT_LABELS[f], f) for f in clustered_features(data, c)]

    def show_report(*_):
        with plot_area:
            plot_area.clear_output()
            plot_cluster_report(data, cell_menu.value, feat_menu.value)

    def switch_cell(change):
        # a cell's feature list may keep the same selected value; redraw explicitly, once
        feat_menu.unobserve(show_report, names='value')
        feat_menu.options = feature_options(change['new'])
        feat_menu.observe(show_report, names='value')
        show_report()

    feat_menu.options = feature_options(cell)
    feat_menu.value = feature
    cell_menu.observe(switch_cell, names='value')
    feat_menu.observe(show_report, names='value')
    display(widgets.HBox([cell_menu, feat_menu]), plot_area)
    show_report()
