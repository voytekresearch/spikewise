"""
config.py
---------
Dataset-wide constants for the Allen Cell Types dataset.

The Allen Cell Types Database provides precomputed electrophysiology features
from intracellular patch-clamp recordings (whole-cell current clamp) of mouse
and human neurons. We use these precomputed features to run the same spike
waveform clustering and population analyses applied to spe-1 and pvc-6.

Data access: allensdk CellTypesCache
Reference: https://alleninstitute.github.io/AllenSDK/cell_types.html

Sections:
  - Cache / data paths (override via environment variables)
  - Species / region filters
  - Feature column definitions (precomputed ephys features of interest)
  - Metadata column definitions
"""

import os

# =============================================================================
# CACHE / DATA PATHS
# =============================================================================
# AllenSDK downloads data here. Override without editing via:
#
#   export ALLEN_CT_CACHE_DIR="/path/to/allen_cell_types_cache"
#   export ALLEN_CT_PICKLE_ROOT="/path/to/allen_ct_pickles"
#
ALLEN_CT_CACHE_DIR = os.environ.get(
    "ALLEN_CT_CACHE_DIR",
    "/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/allen_cell_types_cache",
)
ALLEN_CT_PICKLE_ROOT = os.environ.get(
    "ALLEN_CT_PICKLE_ROOT",
    "/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/allen_ct_pickles",
)

MANIFEST_FILE = os.path.join(ALLEN_CT_CACHE_DIR, "manifest.json")

# =============================================================================
# DATASET FILTERS
# =============================================================================
# Species options: "Mus musculus", "Homo sapiens"
# Set to None to include both.
DEFAULT_SPECIES = "Mus musculus"

# Brain structure acronyms to include. None = all structures.
# Common options: "VISp" (primary visual cortex), "MOp", "SSp", etc.
DEFAULT_STRUCTURES = None  # include all

# =============================================================================
# PRECOMPUTED EPHYS FEATURES OF INTEREST
# =============================================================================
# These are the Allen Cell Types Database precomputed ephys feature columns
# that most directly parallel the spike waveform features extracted in spe-1
# (peak amplitude, threshold, width, upstroke/downstroke, decay).
#
# Stimulus type suffixes:
#   _long_square  — sustained step current (best for waveform shape)
#   _short_square — brief current pulse
#   _ramp         — linearly increasing current

SPIKE_WAVEFORM_FEATURES = [
    # --- Threshold / initiation ---
    "threshold_v_long_square",               # spike threshold voltage (mV)  ~ inflection_mv in spe-1
    "threshold_i_long_square",               # threshold current (pA)

    # --- Peak ---
    "peak_v_long_square",                    # peak voltage (mV)             ~ peak_amplitude in spe-1
    "peak_t_long_square",                    # time to peak (ms)

    # --- Trough / repolarization ---
    "fast_trough_v_long_square",             # fast trough voltage (mV)
    "fast_trough_t_long_square",             # time to fast trough (ms)
    "slow_trough_v_long_square",             # slow trough voltage (mV)

    # --- Waveform shape ---
    "upstroke_downstroke_ratio_long_square", # upstroke/downstroke ratio  ~ peak_sharpness proxy

    # --- Adaptation / firing ---
    "adaptation",                            # inter-spike interval adaptation
    "avg_isi",                               # average inter-spike interval (ms)
    "f_i_curve_slope",                       # slope of f-I curve (Hz/pA)
    "latency",                               # latency to first spike (ms)

    # --- Input resistance / membrane ---
    "input_resistance_mohm",                 # input resistance (MΩ)
    "tau",                                   # membrane time constant (ms)
    "vrest",                                 # resting membrane potential (mV)
    "sag",                                   # sag ratio (Ih current proxy)
]

# Subset most analogous to spe-1 spike features for primary analysis
PRIMARY_SPIKE_FEATURES = [
    "threshold_v_long_square",
    "peak_v_long_square",
    "fast_trough_v_long_square",
    "upstroke_downstroke_ratio_long_square",
    "adaptation",
    "avg_isi",
    "f_i_curve_slope",
]

# =============================================================================
# CELL METADATA COLUMNS
# =============================================================================
METADATA_COLS = [
    "id",                        # specimen ID
    "name",                      # specimen name
    "species",                   # "Mus musculus" or "Homo sapiens"
    "structure_acronym",         # brain area (e.g., "VISp")
    "structure_layer",           # cortical layer
    "dendrite_type",             # "spiny", "aspiny", "sparsely spiny"
    "transgenic_line",           # Cre line
    "reporter_status",           # reporter expression
    "cell_soma_location",        # x, y, z soma position
]
