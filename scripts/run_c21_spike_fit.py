"""Run the c21 patch spike detection exactly like the notebook does."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from spikeparam.patch.fit import Spike


PATCH_FS = 50023.89714358
PATCH_PATH = Path(
    "/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/"
    "Neuropixel Paired Recordings/Recordings/filt_patch_recordings/c21_patch.npy"
)


def main() -> None:
    patch_filt = np.load(PATCH_PATH)
    sp = Spike(thresh_amp=4, window_length=(5.0, 5.0), smooth_frac=0.01)
    sp.fit(patch_filt, PATCH_FS, n_jobs=-1, progress=tqdm, flip_signal=False)

    print(f"path={PATCH_PATH}")
    print(f"fs={PATCH_FS}")
    print(f"thresh_ms={sp.thresh_ms}")
    print(f"thresh_amp={sp.thresh_amp}")
    print(f"n_spikes={len(sp.spike_idxs)}")
    if len(sp.spike_idxs):
        print(f"first10={sp.spike_idxs[:10].tolist()}")


if __name__ == "__main__":
    main()
