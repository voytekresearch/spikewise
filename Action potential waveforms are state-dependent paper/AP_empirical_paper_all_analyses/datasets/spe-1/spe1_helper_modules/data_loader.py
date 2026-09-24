"""
data_loader.py
--------------
Loads raw spe-1 binary recordings, filters them, and saves per-cell .npy files
for LFP, Neuropixels, and patch clamp signals.
"""

import os
import numpy as np
from tqdm.notebook import tqdm

try:
    from .config import NPX_CHANNELS, CELL_IDS, DICT_PATCH_FS, DICT_CHAN_PRED, FILTER_SETTINGS
    from .signal_utils import butter_bandpass
except ImportError:
    from config import NPX_CHANNELS, CELL_IDS, DICT_PATCH_FS, DICT_CHAN_PRED, FILTER_SETTINGS
    from signal_utils import butter_bandpass


VALID_FILE_TYPES = {"patch_ch1", "npx_raw", "npx_lfp"}


def _parse_recording_filename(filename: str):
    """Extract the cell number and recording type from a .bin filename."""

    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')
    cell_num = int(parts[0][1:])
    file_type = '_'.join(parts[1:])

    return cell_num, file_type


def _times_in_ms(n_samples: int, fs: float) -> np.ndarray:
    """Generate sample times in milliseconds."""

    return np.arange(n_samples) / (fs / 1000)


def load_spe1_data(
    data_path: str,
    out_lfp_path: str,
    out_npx_path: str,
    out_patch_path: str,
    *,
    filter_lfp: bool = False,
    verbose: bool = True,
):
    """
    Load, filter, and save spe-1 binary recordings to .npy files.

    Scans data_path for .bin files named as 'cN_<type>.bin' (e.g. 'c21_npx_lfp.bin'),
    parses the cell number and file type, filters each signal, and saves the result.

    Parameters
    ----------
    data_path : str
        Directory containing raw .bin recording files.
    out_lfp_path : str
        Output directory for filtered LFP .npy files.
    out_npx_path : str
        Output directory for filtered Neuropixels .npy files.
    out_patch_path : str
        Output directory for filtered patch clamp .npy files.
    filter_lfp : bool, optional, default: False
        Apply the configured LFP bandpass filter before saving if True.
        Defaults to False to preserve the historical behavior of saving raw LFP traces.
    verbose : bool, optional, default: True
        Print skipped-file diagnostics if True.

    Returns
    -------
    dict with keys 'lfp', 'patch', 'npx', each containing a tuple of
    (list of filtered arrays, list of time arrays in ms).
    """
    # Create output directories
    os.makedirs(out_lfp_path, exist_ok=True)
    os.makedirs(out_npx_path, exist_ok=True)
    os.makedirs(out_patch_path, exist_ok=True)

    # Initialize return values
    lfp_data, patch_data, npx_data = [], [], []
    lfp_times, patch_times, npx_times = [], [], []


    for filename in tqdm(sorted(os.listdir(data_path))):

        if not filename.endswith('.bin'):
            continue

        try:
            cell_num, file_type = _parse_recording_filename(filename)
        except (IndexError, ValueError) as e:
            if verbose:
                print(f"Skipping {filename}: invalid format ({str(e)})")
            continue

        # Validate cell
        if f"c{cell_num}" not in CELL_IDS:
            if verbose:
                print(f"Skipping {filename}: unknown cell id c{cell_num}")
            continue
        if cell_num not in DICT_CHAN_PRED:
            if verbose:
                print(f"Skipping {filename}: no channel mapping for cell {cell_num}")
            continue
        if cell_num not in DICT_PATCH_FS:
            if verbose:
                print(f"Skipping {filename}: no sampling rate metadata for cell {cell_num}")
            continue
        if file_type not in VALID_FILE_TYPES:
            if verbose:
                print(f"Skipping {filename}: unsupported recording type '{file_type}'")
            continue

        # Get parameters for this cell
        channel = DICT_CHAN_PRED[cell_num]
        fs = DICT_PATCH_FS[cell_num]

        # Load and process data
        file_path = os.path.join(data_path, filename)

        if file_type == "patch_ch1":
            data = np.fromfile(file_path, dtype='float64')
            times = _times_in_ms(len(data), fs)
            filtered = butter_bandpass(data, fs, FILTER_SETTINGS['patch'])
            np.save(os.path.join(out_patch_path, f'c{cell_num}_patch.npy'), filtered)
            patch_data.append(filtered)
            patch_times.append(times)

        elif file_type == "npx_raw":
            data = np.memmap(file_path, dtype=np.int16, mode='r')
            data = data.reshape((NPX_CHANNELS, -1), order='F')[channel, :]
            times = _times_in_ms(len(data), fs)
            filtered = butter_bandpass(data, fs, FILTER_SETTINGS['npx'])
            np.save(os.path.join(out_npx_path, f'c{cell_num}_npx.npy'), filtered)
            npx_data.append(filtered)
            npx_times.append(times)

        elif file_type == "npx_lfp":
            data = np.memmap(file_path, dtype=np.int16, mode='r')
            data = data.reshape((-1, NPX_CHANNELS), order='F')[:, channel]
            times = _times_in_ms(len(data), fs)
            filtered = butter_bandpass(data, fs, FILTER_SETTINGS['lfp']) if filter_lfp else data
            np.save(os.path.join(out_lfp_path, f'c{cell_num}_lfp.npy'), filtered)
            lfp_data.append(filtered)
            lfp_times.append(times)

       
    return {
        'lfp': (lfp_data, lfp_times),
        'patch': (patch_data, patch_times),
        'npx': (npx_data, npx_times)
    }
