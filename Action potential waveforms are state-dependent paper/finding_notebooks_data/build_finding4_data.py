"""Extract the small, self-contained data used by 4_within_neuron_variability_exceeds_between_neuron_differences.ipynb.

Only needs to be re-run by maintainers who have the original spe-1 pickles locally.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'AP_empirical_paper_all_analyses', 'datasets', 'spe-1', 'spe1_helper_modules'))
import config
from spk_feat_cluster_comp_analysis import compile_experiment_results, load_spike_fit_pickle

CLUSTER_PKL_DIR = os.path.join(config.SPE1_PICKLE_ROOT, 'cluster_pickles')
SPIKE_FIT_DIR = os.path.join(config.SPE1_PICKLE_ROOT, 'spike_fit_pickles')
OUT = os.path.join(HERE, 'finding4')
os.makedirs(OUT, exist_ok=True)

HALF_WIN = 75        # same +/- 75 samples around the cell-mean peak as the paper's distance functions
N_EXAMPLES = 200     # random single spikes kept per cell for the explorer
EMPTY = np.array([], dtype=np.float32)
rng = np.random.default_rng(0)

# 1. Cell metadata (the columns of the paper's df_master that the distance functions use)
df_master = compile_experiment_results(CLUSTER_PKL_DIR)
meta = df_master[['cell_id', 'cell_type', 'patch_type', 'current_type']].drop_duplicates('cell_id')
meta.to_csv(os.path.join(OUT, 'cells.csv'), index=False)

# 2. Spike-to-average distance cache read by plot_spike_to_avg_distances (Supp Fig 14A/B).
#    Only nRMSE (plotted) and the within / all-cells cosine similarity (reported in the text) are kept;
#    the other keys are left empty so the paper function loads the file unchanged.
v2 = np.load(os.path.join(CLUSTER_PKL_DIR, '_spike_to_avg_distances_v2.npz'))
keep = {k: v2[k].astype(np.float32) for k in v2.files if k.endswith('_nrmse')}
keep.update(within_cos=v2['within_cos'].astype(np.float32),
            **{'Between_(all)_cos': v2['Between_(all)_cos'].astype(np.float32)})
for k in v2.files:
    if k not in keep and k != 'has_wc':
        keep[k] = EMPTY
keep['has_wc'] = v2['has_wc']
np.savez_compressed(os.path.join(OUT, '_spike_to_avg_distances_v2.npz'), **keep)

# 3. Per-cell single-spike nRMSE cache read by plot_waveform_dist_sorted_dots (Fig 4D); RMSE left empty
sd = np.load(os.path.join(CLUSTER_PKL_DIR, '_sorted_dots_v2.npz'))
cell_ids = [str(c) for c in sd['cell_ids']]
dots = {'cell_ids': np.array(cell_ids), 'btw_nrmse_all': sd['btw_nrmse_all'].astype(np.float32),
        'btw_rmse_all': EMPTY}
for i in range(len(cell_ids)):
    dots[f'nrmse_{i}'] = sd[f'nrmse_{i}'].astype(np.float32)
    dots[f'rmse_{i}'] = EMPTY
np.savez_compressed(os.path.join(OUT, '_sorted_dots_v2.npz'), **dots)

# 4. For the explorer: each cell's mean waveform (all spikes) and a random sample of its single spikes,
#    cut to the same window as the paper's nRMSE
examples = {}
for cid in cell_ids:
    sp = load_spike_fit_pickle(os.path.join(SPIKE_FIT_DIR, f'{cid}_spike_fit.pkl'))
    W = np.asarray(sp.spikes, float)
    avg = W.mean(axis=0)
    pk = int(np.argmax(np.abs(avg)))
    lo, hi = pk - HALF_WIN, pk + HALF_WIN
    idx = np.sort(rng.choice(len(W), size=min(N_EXAMPLES, len(W)), replace=False))
    examples[f'{cid}__mean'] = avg[lo:hi].astype(np.float32)
    examples[f'{cid}__spikes'] = W[idx, lo:hi].astype(np.float16)
    examples[f'{cid}__fs'] = np.float64(sp.fs)
    del sp, W
    print(cid)
np.savez_compressed(os.path.join(OUT, 'cell_waveforms.npz'), **examples)

print('wrote', sorted(os.listdir(OUT)))
