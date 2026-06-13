"""
data_loader.py
--------------
Downloads Allen Cell Types NWB files and extracts raw voltage traces
for processing through the spikeparam pipeline.

Analogy to spe-1:
  spe-1 data_loader.py  →  loads .bin files, extracts patch voltage traces
  this data_loader.py   →  downloads NWB files, extracts voltage traces from
                           Long Square sweeps (the Allen analog of spe-1's
                           constant-step patch recordings)

The extracted traces are then passed through the same spikeparam spike
detection and waveform extraction pipeline used for spe-1.

Stimulus mapping:
  Allen "Long Square"   ≈  spe-1 constant current injection
  Allen "Short Square"  ≈  spe-1 brief step
  Allen "Ramp"          ≈  spe-1 ramp stimulus

Usage
-----
    from allen_ct_helper_modules.data_loader import (
        load_cell_metadata, get_long_square_sweeps, load_voltage_trace
    )

    cells_df = load_cell_metadata()
    specimen_id = cells_df.iloc[0]["id"]
    sweeps = get_long_square_sweeps(specimen_id)
    voltage, stimulus, times, fs = load_voltage_trace(specimen_id, sweeps[0])
"""

import os
import numpy as np
import pandas as pd

try:
    from .config import (
        MANIFEST_FILE, ALLEN_CT_PICKLE_ROOT,
        DEFAULT_SPECIES, DEFAULT_STRUCTURES,
    )
except ImportError:
    from config import (
        MANIFEST_FILE, ALLEN_CT_PICKLE_ROOT,
        DEFAULT_SPECIES, DEFAULT_STRUCTURES,
    )

# Stimulus names in Allen NWB files that correspond to sustained current steps
# (most directly analogous to spe-1's constant injection sweeps)
LONG_SQUARE_STIM_NAMES = ["Long Square", "Long Square Threshold"]
SHORT_SQUARE_STIM_NAMES = ["Short Square", "Short Square Threshold"]
RAMP_STIM_NAMES = ["Ramp"]


def _get_cache():
    """Initialize and return a CellTypesCache instance."""
    from allensdk.core.cell_types_cache import CellTypesCache
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    return CellTypesCache(manifest_file=MANIFEST_FILE)


# =============================================================================
# CELL METADATA
# =============================================================================

def load_cell_metadata(
    species: str = DEFAULT_SPECIES,
    structures: list = DEFAULT_STRUCTURES,
    require_reconstruction: bool = False,
) -> pd.DataFrame:
    """
    Download and return cell metadata.

    Parameters
    ----------
    species : str or None
        Filter by species (e.g. "Mus musculus"). None = all species.
    structures : list or None
        Filter by brain structure acronyms (e.g. ["VISp"]). None = all.
    require_reconstruction : bool
        Restrict to cells with morphological reconstructions.

    Returns
    -------
    pd.DataFrame
        One row per cell. Key columns: id, species, structure_acronym,
        structure_layer, dendrite_type, transgenic_line.
    """
    ctc = _get_cache()
    cells = ctc.get_cells(require_reconstruction=require_reconstruction)
    df = pd.DataFrame(cells)

    if species is not None and "species" in df.columns:
        df = df[df["species"] == species].copy()

    if structures is not None and "structure_acronym" in df.columns:
        df = df[df["structure_acronym"].isin(structures)].copy()

    df = df.reset_index(drop=True)
    print(f"Loaded metadata for {len(df)} cells "
          f"(species={species}, structures={structures})")
    return df


# =============================================================================
# SWEEP SELECTION
# =============================================================================

def get_sweeps(specimen_id: int, stim_names: list) -> list:
    """
    Return sweep metadata for a given cell and stimulus type(s).

    Parameters
    ----------
    specimen_id : int
        Allen specimen ID.
    stim_names : list of str
        Stimulus name(s) to select (see LONG_SQUARE_STIM_NAMES, etc.).

    Returns
    -------
    list of dict, one per matching sweep, sorted by sweep number.
    """
    ctc = _get_cache()
    all_sweeps = ctc.get_ephys_sweeps(specimen_id)
    matching = [s for s in all_sweeps
                if s.get("stimulus_name") in stim_names]
    return sorted(matching, key=lambda s: s["sweep_number"])


def get_long_square_sweeps(specimen_id: int) -> list:
    """Return Long Square sweep metadata for a cell (spe-1 analog: constant step)."""
    return get_sweeps(specimen_id, LONG_SQUARE_STIM_NAMES)


def get_ramp_sweeps(specimen_id: int) -> list:
    """Return Ramp sweep metadata for a cell."""
    return get_sweeps(specimen_id, RAMP_STIM_NAMES)


# =============================================================================
# RAW TRACE EXTRACTION
# =============================================================================

def load_voltage_trace(
    specimen_id: int,
    sweep_meta: dict,
) -> tuple:
    """
    Load the raw voltage trace and stimulus for one sweep.

    This is the Allen CT analog of spe-1's patch channel loading. The returned
    voltage array is in the same format as spe-1's patch .npy files — a 1D
    numpy array in mV at the sweep's sampling rate.

    Parameters
    ----------
    specimen_id : int
        Allen specimen ID.
    sweep_meta : dict
        Sweep metadata dict from get_long_square_sweeps() or get_sweeps().

    Returns
    -------
    voltage : np.ndarray
        Voltage trace in mV (1D, shape: [n_samples]).
    stimulus : np.ndarray
        Stimulus current in pA (1D, shape: [n_samples]).
    times : np.ndarray
        Time array in ms (1D, shape: [n_samples]).
    fs : float
        Sampling rate in Hz.
    index_range : tuple of (int, int)
        Start/stop sample indices excluding the test pulse.
    """
    ctc = _get_cache()
    data_set = ctc.get_ephys_data(specimen_id)

    sweep_number = sweep_meta["sweep_number"]
    sweep_data = data_set.get_sweep(sweep_number)

    # Voltage in V → convert to mV (same units as spe-1 patch)
    voltage = sweep_data["response"] * 1e3  # V → mV
    stimulus = sweep_data["stimulus"] * 1e12  # A → pA
    fs = sweep_data["sampling_rate"]
    index_range = sweep_data["index_range"]

    # Times in ms (same as spe-1 convention)
    times = np.arange(len(voltage)) / (fs / 1000.0)

    return voltage, stimulus, times, fs, index_range


def load_all_long_square_traces(
    specimen_id: int,
    trim_test_pulse: bool = True,
) -> list:
    """
    Load all Long Square voltage traces for a cell.

    Analogous to loading all constant-step sweeps from spe-1's patch file.

    Parameters
    ----------
    specimen_id : int
    trim_test_pulse : bool
        If True, trim traces to index_range (excludes the test pulse).

    Returns
    -------
    list of dicts, each containing:
        'sweep_number', 'voltage' (mV), 'stimulus' (pA), 'times' (ms),
        'fs' (Hz), 'amplitude_pa' (step amplitude from sweep metadata).
    """
    sweeps_meta = get_long_square_sweeps(specimen_id)
    results = []

    for s in sweeps_meta:
        try:
            voltage, stimulus, times, fs, idx_range = load_voltage_trace(
                specimen_id, s
            )
            if trim_test_pulse and idx_range is not None:
                start, stop = idx_range
                voltage = voltage[start:stop]
                stimulus = stimulus[start:stop]
                times = times[start:stop]

            results.append({
                "sweep_number": s["sweep_number"],
                "voltage": voltage,
                "stimulus": stimulus,
                "times": times,
                "fs": fs,
                "amplitude_pa": s.get("stimulus_amplitude", np.nan),
            })
        except Exception as e:
            print(f"  Skipping sweep {s['sweep_number']} for specimen "
                  f"{specimen_id}: {e}")
            continue

    return results


# =============================================================================
# BATCH LOADING (POPULATION)
# =============================================================================

def load_population_traces(
    cells_df: pd.DataFrame,
    n_cells: int = None,
    trim_test_pulse: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Load Long Square voltage traces for a population of cells.

    This is the population-level analog of spe-1's load_spe1_data(), where
    all cells' patch recordings are loaded in one pass. The returned dict
    feeds into the same spikeparam spike detection and waveform extraction
    pipeline.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Output of load_cell_metadata().
    n_cells : int or None
        If set, only load the first N cells (useful for testing).
    trim_test_pulse : bool
        Trim traces to exclude the test pulse epoch.
    verbose : bool
        Print progress.

    Returns
    -------
    dict mapping specimen_id (int) → list of sweep dicts
        (same format as load_all_long_square_traces output).
    """
    if n_cells is not None:
        cells_df = cells_df.iloc[:n_cells]

    population = {}
    for i, row in cells_df.iterrows():
        sid = int(row["id"])
        if verbose:
            print(f"[{i+1}/{len(cells_df)}] Loading specimen {sid} "
                  f"({row.get('structure_acronym', '?')})")
        try:
            sweeps = load_all_long_square_traces(
                sid, trim_test_pulse=trim_test_pulse
            )
            population[sid] = sweeps
            if verbose:
                print(f"  → {len(sweeps)} Long Square sweeps")
        except Exception as e:
            if verbose:
                print(f"  → FAILED: {e}")
            population[sid] = []

    n_loaded = sum(1 for v in population.values() if len(v) > 0)
    print(f"\nLoaded traces for {n_loaded}/{len(cells_df)} cells.")
    return population


# =============================================================================
# PICKLE UTILITIES (mirrors spe-1 pattern)
# =============================================================================

def save_to_pickle(obj, filename: str, subdir: str = "") -> str:
    """Save obj to ALLEN_CT_PICKLE_ROOT/subdir/filename.pkl"""
    import pickle
    save_dir = os.path.join(ALLEN_CT_PICKLE_ROOT, subdir) if subdir else ALLEN_CT_PICKLE_ROOT
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{filename}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved → {path}")
    return path


def load_from_pickle(filename: str, subdir: str = ""):
    """Load from ALLEN_CT_PICKLE_ROOT/subdir/filename.pkl"""
    import pickle
    load_dir = os.path.join(ALLEN_CT_PICKLE_ROOT, subdir) if subdir else ALLEN_CT_PICKLE_ROOT
    path = os.path.join(load_dir, f"{filename}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
