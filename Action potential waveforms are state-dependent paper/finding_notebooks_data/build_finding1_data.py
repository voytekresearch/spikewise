"""Extract the small, self-contained data used by 1_parameterization_captures_intra_spike_correlations.ipynb.

Only needs to be re-run by maintainers who have the original raw data and pickles locally.
"""
import os
import sys
import pickle

import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'AP_empirical_paper_all_analyses', 'datasets', 'spe-1', 'spe1_helper_modules'))
import config

PVC6_H5 = '/Users/blancamartin/Downloads/153098.04.02_data.h5'
PVC6_PKL = '/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/pvc6_pickles/'
CLUSTER_PKL_DIR = os.path.join(config.SPE1_PICKLE_ROOT, 'cluster_pickles')
OUT = os.path.join(HERE, 'finding1')
os.makedirs(OUT, exist_ok=True)

SPK_FEATS = ['ramp_amp', 'inflection_time', 'inflection_amp',
             'peak_amp', 'peak_width', 'peak_sharpness',
             'exp_lambda', 'exp_const', 'log_isi']
ONE_MS = 200000 // 1000

# Toy spike (Fig 2A): one ~20 ms voltage snippet from pvc-6 sweep 14, same window as
# AP_empirical_algorithm_sch.ipynb (the toy fit is run live in the notebook)
toy_path = os.path.join(OUT, 'pvc6_c1_toy_trace.npy')
if not os.path.exists(toy_path):
    with h5py.File(PVC6_H5, 'r') as f:
        data = f['Sweep_14'][:, 1]
    idx = find_peaks(data, height=-10, distance=ONE_MS)[0]
    np.save(toy_path, data[idx[1] - ONE_MS * 10: idx[2] + ONE_MS * 10].astype(np.float32))

# pvc-6 Cell 1 spikewise features + fit quality for all 820 spikes (Fig 2C table, Fig 2D matrix).
# Cached output of Spike(thresh_amp=-10, window_length=(5., 5.), smooth_frac=.008, pre_inflection_ms=0.5);
# reproduces the Fig 2D correlations exactly.
with open(PVC6_PKL + 'df_all_c1.pkl', 'rb') as f:
    df_c1 = pickle.load(f)
with open(PVC6_PKL + '_r2_c1.pkl', 'rb') as f:
    r2_c1 = pickle.load(f)
df_c1 = df_c1[SPK_FEATS].copy()
df_c1['r_squared_ramp'] = r2_c1['r_squared_ramp']
df_c1['r_squared_exp'] = r2_c1['r_squared_exp']
df_c1.to_csv(os.path.join(OUT, 'pvc6_c1_features.csv.gz'), index=False, float_format='%.6g')

# Per-cell features for the cell picker (Supp Fig 1 inputs)
rows = []
for cell, label in [(1, 'SST+'), (2, 'unmarked')]:
    with open(PVC6_PKL + f'df_all_c{cell}.pkl', 'rb') as f:
        df = pickle.load(f)[SPK_FEATS].copy()
    df.insert(0, 'cell_type', label)
    df.insert(0, 'cell', cell)
    df.insert(0, 'dataset', 'pvc-6')
    rows.append(df)
for cid in config.CELL_IDS:
    with open(os.path.join(CLUSTER_PKL_DIR, f'{cid}_cluster_df.pkl'), 'rb') as f:
        df = pickle.load(f)[SPK_FEATS].copy()
    cnum = int(cid[1:])
    df.insert(0, 'cell_type', {'PC': 'putative PC', 'IN': 'putative IN'}[config.DICT_CELL_TYPE[cnum]])
    df.insert(0, 'cell', cnum)
    df.insert(0, 'dataset', 'spe-1')
    rows.append(df)
pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT, 'per_cell_features.csv.gz'), index=False, float_format='%.6g')

print('wrote', sorted(os.listdir(OUT)))
