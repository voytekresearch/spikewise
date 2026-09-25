"""Build the small example recordings used by the spikewise tutorials.

Only needs to be re-run by maintainers who have the original pvc-6 and spe-1 recordings locally.
Output: example_recordings.npz (pvc-6 sweeps + a short spe-1 juxtacellular recording).
"""
import os

import h5py
import numpy as np
import pandas as pd
from scipy.signal import decimate, resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
PVC6_H5 = '/Users/blancamartin/Downloads/153098.04.02_data.h5'   # pvc-6 Cell 1 (Berg, 2014; CRCNS)
PVC6_FS = 200000
SPE1_PATCH = ('/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/Neuropixel Paired Recordings/'
              'Recordings/filt_patch_recordings/c14_patch.npy')      # spe-1 cell 14 (Marques-Smith et al., 2018)
SPE1_FS = 50023.8986874656
SPE1_SEGMENT_S = (20.0, 25.0)

SWEEPS = {'constant': [11, 20], 'ramp': [0, 1],
          'pink': [13, 14, 15, 17, 18, 19, 21, 22, 23, 28]}
V_FS, I_FS = 50000, 10000          # stored sampling rates: voltage 50 kHz, injected current 10 kHz
V_SCALE, I_SCALE = 100, 100        # stored as int16 in units of 0.01 mV and 0.01 pA
PAD_S = 0.3                        # keep 300 ms before and after the stimulus


def stim_epoch(stim):
    # skip the first 100 ms, which holds a brief test pulse on every sweep
    on = np.flatnonzero(np.abs(stim[PVC6_FS // 10:]) > 1e-3) + PVC6_FS // 10
    pad = int(PAD_S * PVC6_FS)
    return max(on[0] - pad, 0), min(on[-1] + pad, len(stim))


out = {}
rows = []
with h5py.File(PVC6_H5, 'r') as f:
    for stim_type, sweeps in SWEEPS.items():
        for sweep in sweeps:
            dset = f[f'Sweep_{sweep}']
            stim, volt = np.asarray(dset[:, 0]), np.asarray(dset[:, 1])
            lo, hi = stim_epoch(stim)
            v = decimate(volt[lo:hi], PVC6_FS // V_FS, ftype='fir', zero_phase=True)
            i = resample_poly(stim[lo:hi], 1, PVC6_FS // I_FS)
            key = f'sweep{sweep}'
            out[f'{key}_voltage'] = np.round(v * V_SCALE).astype(np.int16)
            out[f'{key}_current'] = np.round(i * I_SCALE).astype(np.int16)
            rows.append((key, sweep, stim_type))

patch = np.load(SPE1_PATCH, mmap_mode='r')
seg = np.asarray(patch[int(SPE1_SEGMENT_S[0] * SPE1_FS):int(SPE1_SEGMENT_S[1] * SPE1_FS)], float)
out['spe1_juxta'] = np.round(seg * 10).astype(np.int16)   # units of 0.1 (raw patch units)

np.savez_compressed(os.path.join(HERE, 'example_recordings.npz'),
                    sweep_keys=np.array([r[0] for r in rows]), sweep_numbers=np.array([r[1] for r in rows]),
                    stim_types=np.array([r[2] for r in rows]),
                    voltage_fs=V_FS, current_fs=I_FS, voltage_scale=V_SCALE, current_scale=I_SCALE,
                    spe1_fs=SPE1_FS, spe1_scale=10, **out)
print(pd.DataFrame(rows, columns=['key', 'sweep', 'stim_type']).to_string(index=False))
