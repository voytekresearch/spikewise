"""Extract the small, self-contained data used by 2_electrical_stimulation_causally_influences_spike_waveform.ipynb.

Only needs to be re-run by maintainers who have the original raw data and pickles locally.
"""
import gzip
import json
import os
import pickle

import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
PVC6_H5 = '/Users/blancamartin/Downloads/153098.04.02_data.h5'
PVC6_PKL = '/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/pvc6_pickles/'
OUT = os.path.join(HERE, 'finding2')
os.makedirs(OUT, exist_ok=True)

FS = 200000
ONE_MS = FS // 1000
WINDOWS_MS = [5, 25, 50, 100, 200, 300, 400, 500]
EXAMPLE_SWEEPS = {'constant': 20, 'ramp': 0, 'pink': 13}
TRACE_DECIM = 10   # example sweeps stored at 20 kHz

with open(PVC6_PKL + 'df_all_c1.pkl', 'rb') as fh:
    df = pickle.load(fh)
h5 = h5py.File(PVC6_H5, 'r')


def sweep_arrays(sweep):
    dset = h5['Sweep_' + str(int(sweep))]
    return np.asarray(dset[:, 0]), np.asarray(dset[:, 1])


def stim_onset(stim):
    # skip the first 100 ms, which holds a brief test pulse on every sweep
    return np.flatnonzero(np.abs(stim[100 * ONE_MS:]) > 1e-3)[0] + 100 * ONE_MS


def peaks(v):
    # same spike detection as pvc6_stim_analysis.recompute_stim_features
    return find_peaks(v, height=-10, distance=ONE_MS)[0]


# 1. One example sweep per stimulation type, cropped to the stimulus epoch (+/- 300 ms)
traces = {}
for stim_type, sweep in EXAMPLE_SWEEPS.items():
    stim, volt = sweep_arrays(sweep)
    on = np.flatnonzero(np.abs(stim[100 * ONE_MS:]) > 1e-3) + 100 * ONE_MS
    lo, hi = max(on[0] - 300 * ONE_MS, 0), min(on[-1] + 300 * ONE_MS, len(stim))
    traces[f'{stim_type}_stim'] = stim[lo:hi:TRACE_DECIM].astype(np.float32)
    traces[f'{stim_type}_volt'] = volt[lo:hi:TRACE_DECIM].astype(np.float32)
    traces[f'{stim_type}_sweep'] = sweep
np.savez_compressed(os.path.join(OUT, 'example_sweeps.npz'), fs=FS // TRACE_DECIM, **traces)

# 2. One pink-noise spike with its full-resolution pre-spike stimulus (for the window schematic)
pink = df[(df['stim_type'] == 'pink') & (df['sweep'] == EXAMPLE_SWEEPS['pink'])].sort_values('spike_num')
stim, volt = sweep_arrays(EXAMPLE_SWEEPS['pink'])
pk = peaks(volt)
onset = stim_onset(stim)
for _, row in pink.iterrows():
    n = int(row['spike_num'])
    p = pk[n]
    infl = p - int(row['inflection_time'] * ONE_MS)
    prev_ok = n == 0 or (p - pk[n - 1]) > 550 * ONE_MS  # no earlier spike inside the longest window
    if prev_ok and infl - 550 * ONE_MS >= onset:   # pink noise already on for the whole longest window
        lo, hi = infl - 550 * ONE_MS, p + 20 * ONE_MS
        np.savez_compressed(os.path.join(OUT, 'pink_example_spike.npz'), fs=FS,
                            stim=stim[lo:hi].astype(np.float32), volt=volt[lo:hi].astype(np.float32),
                            inflection_idx=infl - lo, peak_idx=p - lo)
        break
else:
    raise RuntimeError('no isolated pink-noise spike found')

# 3. Pink-noise spikes: waveform features + pre-spike stimulus features for every window size
with open(PVC6_PKL + 'df_pink_filtered_c1.pkl', 'rb') as fh:
    df_pink = pickle.load(fh)
WAVE_FEATS = ['ramp_amp', 'inflection_time', 'inflection_amp', 'peak_amp', 'peak_width',
              'peak_sharpness', 'exp_lambda', 'exp_const', 'log_isi']
stim_feats = df_pink[WAVE_FEATS].copy()
for w in WINDOWS_MS:
    if w == 5:
        extra = df_pink[['stim_mean', 'stim_std', 'stim_exp']].add_suffix('_5ms')
    else:
        with open(PVC6_PKL + f'_stim_extra_{w}ms.pkl', 'rb') as fh:
            extra = pickle.load(fh)
    stim_feats = stim_feats.join(extra[[f'stim_mean_{w}ms', f'stim_std_{w}ms', f'stim_exp_{w}ms']])
stim_feats.to_csv(os.path.join(OUT, 'pink_spikes_by_window.csv.gz'), index=False, float_format='%.6g')

# 4. Stimulation type classifier (Fig 3B/C): repeated out-of-fold results
with open(PVC6_PKL + 'rf_model.pkl', 'rb') as fh:
    rf = pickle.load(fh)
with open(PVC6_PKL + 'rf_repeated_oof.pkl', 'rb') as fh:
    rep_acc, rep_cms, rep_imp = pickle.load(fh)
np.savez_compressed(os.path.join(OUT, 'stim_type_classifier.npz'),
                    accuracies=rep_acc, mean_confusion=rep_cms.mean(axis=0), importances=rep_imp,
                    classes=np.array(rf.classes_, dtype=str), feature_names=np.array(rf.feature_names_in_, dtype=str))

# 5. Ridge regression results for every window (Fig 3E-G, Supp window expansion)
KEEP = ['y_true', 'y_pred_cv', 'coefficients', 'feature_names', 'ci_lower', 'ci_upper',
        'p_values', 'r2_mean', 'r2_ci', 'p_val_perm', 'bootstrapped_r2']
with open(PVC6_PKL + 'results_window.pkl', 'rb') as fh:
    rw = pickle.load(fh)
ridge = {k: {f: np.asarray(v[f]).tolist() for f in KEEP} for k, v in rw.items()}
with gzip.open(os.path.join(OUT, 'ridge_results_by_window.json.gz'), 'wt') as fh:
    json.dump(ridge, fh)

print('wrote', sorted(os.listdir(OUT)))
