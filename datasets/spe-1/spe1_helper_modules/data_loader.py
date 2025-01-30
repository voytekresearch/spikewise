# data_loader.py

import os
import numpy as np
from tqdm.notebook import tqdm
from config import NPX_CHANNELS, CELL_IDS, DICT_PATCH_FS, DICT_CHAN_PRED, FILTER_SETTINGS
from signal_utils import butter_bandpass

def load_spe1_data(data_path: str, out_lfp_path: str, out_npx_path: str, out_patch_path: str):
    """Main data loading function with corrected filename parsing."""
    # Create output directories
    os.makedirs(out_lfp_path, exist_ok=True)
    os.makedirs(out_npx_path, exist_ok=True)
    os.makedirs(out_patch_path, exist_ok=True)

    # Initialize return values
    lfp_data, patch_data, npx_data = [], [], []
    lfp_times, patch_times, npx_times = [], [], []


    for filename in tqdm(os.listdir(data_path)):
        if not filename.endswith('.bin'):
            continue

        try:
            # Split filename into parts (e.g., "c6_npx_raw.bin" -> ["c6", "npx", "raw"])
            basename = os.path.splitext(filename)[0]  # Remove ".bin"
            parts = basename.split('_')
            cell_num = int(parts[0][1:])  # Extract number from "c6", "c14", etc.
            file_type = '_'.join(parts[1:])  # "npx_raw", "patch_ch1", etc.
        except (IndexError, ValueError) as e:
            print(f"Skipping {filename}: invalid format ({str(e)})")
            continue

        # Validate cell
        if f"c{cell_num}" not in CELL_IDS:
            continue
        if cell_num not in DICT_CHAN_PRED:
            print(f"Skipping cell {cell_num}: no channel mapping")
            continue

        # Rest of the code remains the same...
        # (loading, filtering, saving logic)
        # Validate cell
        if f"c{cell_num}" not in CELL_IDS:
            continue
        if cell_num not in DICT_CHAN_PRED:
            print(f"Skipping cell {cell_num}: no channel mapping")
            continue

        # Inside the loop after parsing:

        # Get parameters for this cell
        channel = DICT_CHAN_PRED[cell_num]
        fs = DICT_PATCH_FS[cell_num]

        # Load and process data
        file_path = os.path.join(data_path, filename)

        if file_type == "patch_ch1":
            # Process patch data (unchanged)
            data = np.fromfile(file_path, dtype='float64')
            times = np.arange(len(data)) / (fs / 1000)  # ms
            filtered = butter_bandpass(data, fs, FILTER_SETTINGS['patch'])
            np.save(os.path.join(out_patch_path, f'c{cell_num}_patch.npy'), filtered)
            patch_data.append(filtered)
            patch_times.append(times)

        elif file_type == "npx_raw":
            # Process Neuropixels raw data
            data = np.memmap(file_path, dtype=np.int16, mode='r')
            # Reshape to (channels, samples) and select the channel
            data = data.reshape((NPX_CHANNELS, -1), order='F')[channel, :]
            times = np.arange(len(data)) / (fs / 1000)  # ms
            filtered = butter_bandpass(data, fs, FILTER_SETTINGS['npx'])
            np.save(os.path.join(out_npx_path, f'c{cell_num}_npx.npy'), filtered)
            npx_data.append(filtered)
            npx_times.append(times)

        elif file_type == "npx_lfp":
            # Process LFP data (if needed)
            data = np.memmap(file_path, dtype=np.int16, mode='r')
            data = data.reshape((-1, NPX_CHANNELS), order='F')[:, channel]
            times = np.arange(len(data)) / (fs / 1000)
            filtered = butter_bandpass(data, fs, FILTER_SETTINGS['lfp'])
            np.save(os.path.join(out_lfp_path, f'c{cell_num}_lfp.npy'), filtered)
            lfp_data.append(filtered)
            lfp_times.append(times)

       
    return {
        'lfp': (lfp_data, lfp_times),
        'patch': (patch_data, patch_times),
        'npx': (npx_data, npx_times)
    }

