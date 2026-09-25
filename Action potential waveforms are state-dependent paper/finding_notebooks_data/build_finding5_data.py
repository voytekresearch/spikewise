"""Extract the small, self-contained data used by 5_ap_waveform_predicts_peri_spike_lfp_state.ipynb.

Only needs to be re-run by maintainers who have the original raw data and pickles locally.
"""
import contextlib
import gzip
import io
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'AP_empirical_paper_all_analyses', 'datasets', 'spe-1', 'spe1_helper_modules'))
import config
from pop_ridge_utils import merge_hpf_targets, aggregate_population
from ridge_regression_utils import FEAT_LABELS

OUT = os.path.join(HERE, 'finding5')
os.makedirs(OUT, exist_ok=True)
R = config.SPE1_PICKLE_ROOT
CELL = 14                      # pyramidal cell with a clear extracellular spike on the probe
SEGMENT_S = (20.0, 22.0)       # example stretch of recording (early, so spike times and LFP stay aligned)
EXAMPLE_SPIKE = 100            # spike used for the peri-spike LFP window example
STA_MAX_S = 120.0              # spikes used for the average waveforms


def tolist(d):
    if isinstance(d, dict):
        return {k: tolist(v) for k, v in d.items()}
    return np.asarray(d, float).tolist()


# 1. Population ridge regression results (39 cells): per-cell CV R², significance and betas
with contextlib.redirect_stdout(io.StringIO()):
    all_results, cell_ids, target_names, target_labels, predictor_sets = merge_hpf_targets(
        R + '/ridge_regression_pickles', FEAT_LABELS)
    r2_pop, sig_pop, beta_pop = aggregate_population(all_results, cell_ids, target_names, predictor_sets)
with gzip.open(os.path.join(OUT, 'ridge_population.json.gz'), 'wt') as fh:
    json.dump({'cell_ids': list(cell_ids), 'target_names': list(target_names),
               'target_labels': list(target_labels), 'predictor_sets': list(predictor_sets),
               'cell_types': {c: config.DICT_CELL_TYPE[int(c[1:])] for c in cell_ids},
               'r2_pop': tolist(r2_pop), 'sig_pop': tolist(sig_pop), 'beta_pop': tolist(beta_pop)}, fh)

# 2. Example recording: intracellular (patch), extracellular (Neuropixels) and LFP, same clock
patch_fs, npx_fs, lfp_fs = config.DICT_PATCH_FS[CELL], config.NPX_FS, config.LFP_FS
D = config.SPE1_DATA_ROOT
# juxtacellular signal is negative-going; flipped as in the cell's own analysis (sp.fit(..., flip_signal=True))
patch = -np.load(f'{D}/filt_patch_recordings/c{CELL}_patch.npy', mmap_mode='r')
npx = np.load(f'{D}/filt_npx_recordings/c{CELL}npx_filt.npy', mmap_mode='r')
lfp = np.load(f'{D}/filt_lfp_recordings/c{CELL}_lfp.npy', mmap_mode='r')


def seg(x, fs, t0, t1, step=1):
    return np.asarray(x[int(t0 * fs):int(t1 * fs):step], np.float32)


with open(f'{R}/cluster_pickles/c{CELL}_cluster_df.pkl', 'rb') as fh:
    cdf = pickle.load(fh)
spk_s = cdf['spk_times_ms'].to_numpy() / 1000.0
np.savez_compressed(os.path.join(OUT, 'example_recording.npz'), cell=CELL, t0=SEGMENT_S[0],
                    patch=seg(patch, patch_fs, *SEGMENT_S, step=5), patch_fs=patch_fs / 5,
                    npx=seg(npx, npx_fs, *SEGMENT_S, step=3), npx_fs=npx_fs / 3,
                    lfp=seg(lfp, lfp_fs, *SEGMENT_S), lfp_fs=lfp_fs,
                    spike_times_s=spk_s[(spk_s >= SEGMENT_S[0]) & (spk_s < SEGMENT_S[1])])

# 3. Average intracellular (IAP) and extracellular (EAP) spike, +/- 2 ms around each spike
early = spk_s[(spk_s > 0.01) & (spk_s < STA_MAX_S)]
def sta(x, fs, half_ms=2.0):
    h = int(half_ms / 1000 * fs)
    w = np.array([np.asarray(x[int(t * fs) - h:int(t * fs) + h], float) for t in early])
    return w.mean(0).astype(np.float32), w.std(0).astype(np.float32)
iap_m, iap_s = sta(patch, patch_fs)
eap_m, eap_s = sta(npx, npx_fs)
np.savez_compressed(os.path.join(OUT, 'average_spikes.npz'), n_spikes=len(early),
                    iap_mean=iap_m, iap_sd=iap_s, iap_fs=patch_fs, eap_mean=eap_m, eap_sd=eap_s, eap_fs=npx_fs)

# 4. One spike's surrounding LFP (+/- 1 s) and its stored sliding-window LFP mean / SD
with open(f'{R}/simple_lfp_pickles/c{CELL}_simple_lfp.pkl', 'rb') as fh:
    simple = pickle.load(fh)
t_spk = spk_s[EXAMPLE_SPIKE]
w = simple[EXAMPLE_SPIKE]
np.savez_compressed(os.path.join(OUT, 'example_lfp_window.npz'), lfp_fs=lfp_fs,
                    lfp=seg(lfp, lfp_fs, t_spk - 1.0, t_spk + 1.0),
                    t_bins_s=np.asarray(w['t_bins_s'], np.float32),
                    lfp_mean=np.asarray(w['lfp_mean'], np.float32), lfp_std=np.asarray(w['lfp_std'], np.float32))

print('wrote', sorted(os.listdir(OUT)))
