"""Extract the small, self-contained data used by 3_neurons_show_multimodal_spontaneous_variability.ipynb.

Only needs to be re-run by maintainers who have the original spe-1 pickles locally.
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'AP_empirical_paper_all_analyses', 'datasets', 'spe-1', 'spe1_helper_modules'))
import config
from spk_feat_cluster_comp_analysis import load_spike_fit_pickle

CLUSTER_PKL_DIR = os.path.join(config.SPE1_PICKLE_ROOT, 'cluster_pickles')
SPIKE_FIT_DIR = os.path.join(config.SPE1_PICKLE_ROOT, 'spike_fit_pickles')
OUT = os.path.join(HERE, 'finding3')
os.makedirs(OUT, exist_ok=True)

SPK_FEATS = ['ramp_amp', 'inflection_time', 'inflection_amp',
             'peak_amp', 'peak_width', 'peak_sharpness',
             'exp_lambda', 'exp_const', 'log_isi']
EXCLUDED_CELLS = {'c17', 'c18', 'c43'}   # too few spikes for clustering (n = 40 analyzed)
# Same exclusions as supp_cluster_characterization.ipynb (paper counts: 30/40 cells, 14 peak width, 14 peak amp):
# spike timing is not a waveform feature, exp_const and these two cell x feature pairs are degenerate clusters
SKIP_FEATS = {'spk_times_ms', 'spk_times_idx', 'exp_const'}
SKIP_PAIRS = {('c15', 'peak_width'), ('c7', 'inflection_time')}
HALF_WIN = 250          # cluster-average waveforms kept at +/- 250 samples (+/- 5 ms at ~50 kHz)
EXAMPLE_HALF_WIN = 200  # individual example spikes kept at +/- 200 samples (+/- 4 ms)
N_EXAMPLES = 25         # example spikes per cluster, per clustered feature
rng = np.random.default_rng(0)

cell_ids = sorted((c for c in config.CELL_IDS if c not in EXCLUDED_CELLS), key=lambda c: int(c[1:]))

features, metrics, waveforms, examples, cells = [], [], {}, {}, []
for cid in cell_ids:
    cnum = int(cid[1:])
    df = pd.read_pickle(os.path.join(CLUSTER_PKL_DIR, f'{cid}_cluster_df.pkl'))
    rep = pd.read_pickle(os.path.join(CLUSTER_PKL_DIR, f'{cid}_cluster_report.pkl'))
    # the cluster report lists the final clustered features (a few cluster_df files keep stale extra columns)
    reported = set(rep['feature_clustered']) if 'feature_clustered' in rep.columns else set()
    clust_cols = [f'{f}_cluster' for f in SPK_FEATS
                  if f in reported and f not in SKIP_FEATS and (cid, f) not in SKIP_PAIRS]

    # 1. per-spike features, spike times and cluster labels (Fig 4B, cluster report)
    out = df[['spk_id', 'spk_times_ms'] + SPK_FEATS + clust_cols].copy()
    out.insert(0, 'cell', cnum)
    features.append(out)

    # 2. cluster separation metrics from the per-cell cluster report
    if clust_cols:
        rep = rep[rep['feature_clustered'].isin([c[:-8] for c in clust_cols])]
        metrics.append(rep.drop(columns='cluster_group_id').assign(cell=cnum))

    # 3. cluster-average waveforms (mean +/- SD over all spikes in each cluster; Fig 4C)
    with open(os.path.join(CLUSTER_PKL_DIR, f'{cid}_cluster_waveforms.pkl'), 'rb') as fh:
        wf = pickle.load(fh)
    for col in clust_cols:
        if col not in wf:
            continue
        t = np.asarray(wf[col]['t_axis'])
        keep = (t >= -HALF_WIN) & (t <= HALF_WIN)
        waveforms[f'c{cnum}__{col}__t'] = t[keep].astype(np.int16)
        for lab, v in wf[col].items():
            if lab == 't_axis':
                continue
            waveforms[f'c{cnum}__{col}__{lab}__mean'] = np.asarray(v['mean'])[:len(t)][keep].astype(np.float32)
            waveforms[f'c{cnum}__{col}__{lab}__std'] = np.asarray(v['std'])[:len(t)][keep].astype(np.float32)
            waveforms[f'c{cnum}__{col}__{lab}__n'] = np.int64(v['n'])

    # 4. a few peak-aligned example spikes per cluster, for the individual-spike panel of the cluster report
    ids = set()
    for col in clust_cols:
        for _, g in df.groupby(col):
            ids.update(rng.choice(g['spk_id'].to_numpy(), size=min(N_EXAMPLES, len(g)), replace=False).tolist())
    ids = np.array(sorted(ids), dtype=int)
    if len(ids):
        sp = load_spike_fit_pickle(os.path.join(SPIKE_FIT_DIR, f'{cid}_spike_fit.pkl'))
        spikes = np.full((len(ids), 2 * EXAMPLE_HALF_WIN + 1), np.nan, dtype=np.float32)
        for k, i in enumerate(ids):
            w = np.asarray(sp.spikes[i], dtype=float)
            pk = int(np.argmax(np.abs(w)))
            lo, hi = max(pk - EXAMPLE_HALF_WIN, 0), min(pk + EXAMPLE_HALF_WIN + 1, len(w))
            spikes[k, EXAMPLE_HALF_WIN - (pk - lo):EXAMPLE_HALF_WIN + (hi - pk)] = w[lo:hi]
        examples[f'c{cnum}__spk_id'] = ids
        examples[f'c{cnum}__spikes'] = spikes.astype(np.float16)
        del sp

    cells.append({'cell': cnum, 'cell_type': config.DICT_CELL_TYPE[cnum], 'n_spikes': len(df),
                  'fs': config.DICT_PATCH_FS[cnum]})
    print(cid, len(df), [c[:-8] for c in clust_cols])

pd.concat(features, ignore_index=True).to_csv(os.path.join(OUT, 'cluster_features.csv.gz'), index=False, float_format='%.6g')
pd.concat(metrics, ignore_index=True).to_csv(os.path.join(OUT, 'cluster_metrics.csv'), index=False, float_format='%.6g')
pd.DataFrame(cells).to_csv(os.path.join(OUT, 'cells.csv'), index=False)
np.savez_compressed(os.path.join(OUT, 'cluster_waveforms.npz'), **waveforms)
np.savez_compressed(os.path.join(OUT, 'example_spikes.npz'), **examples)

print('wrote', sorted(os.listdir(OUT)))
