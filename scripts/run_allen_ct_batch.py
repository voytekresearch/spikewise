#!/usr/bin/env python3
"""
run_allen_ct_batch.py
---------------------
Downloads Allen Cell Types NWB files and extracts spikewise spike waveform
features for all cells, saving per-cell pickles.

Pipeline:
  1. Load cell metadata for all mouse cells
  2. For each cell:
     a. Download NWB file (cached locally by AllenSDK)
     b. Find all sweeps (any stimulus type) where num_spikes > 0
     c. For each spiking sweep:
        - Trim to index_range (exclude test pulse)
        - Fit spikewise Spike class on that sweep's voltage
        - Attach sweep metadata (stimulus_name, amplitude, sweep_number) to every spike row
     d. Concatenate all per-sweep spike rows → one DataFrame per cell
     e. Save → allen_ct_pickles/features/{specimen_id}_features.pkl
     f. Save waveforms → allen_ct_pickles/features/{specimen_id}_waveforms.pkl
  3. Build and save population DataFrame
     allen_ct_pickles/allen_ct_population_features.pkl

Output tables (relational, linked by specimen_id / sweep_uid):

  allen_ct_population_spikes.pkl  — one row per spike
      spikewise features + specimen_id + sweep_number + sweep_uid

  allen_ct_sweep_table.pkl        — one row per spiking sweep
      specimen_id, sweep_uid, sweep_number + all Allen sweep metadata
      (stimulus_name, stimulus_absolute_amplitude, num_spikes, …)

  allen_ct_cells_df.pkl           — one row per cell (saved by load_data notebook)
      specimen_id + all Allen cell metadata (dendrite_type, area, layer, …)

Join pattern:
  spikes.merge(sweep_table, on="sweep_uid")
       .merge(cells_df.rename({"id":"specimen_id"}), on="specimen_id")

Usage
-----
    # All cells (default):
    python scripts/run_allen_ct_batch.py

    # Test run — first 10 cells:
    python scripts/run_allen_ct_batch.py --n-cells 10

    # Specific specimen IDs:
    python scripts/run_allen_ct_batch.py --cells 565871768 484679812

    # Force recompute even if pickle already exists:
    python scripts/run_allen_ct_batch.py --force

    # Parallel workers (RAM permitting — each worker uses n_jobs=1 for Spike.fit):
    python scripts/run_allen_ct_batch.py --workers 4

Env vars (override paths without editing):
    ALLEN_CT_CACHE_DIR    — where AllenSDK caches NWB files (default: ~/...//allen_cell_types_cache)
    ALLEN_CT_PICKLE_ROOT  — where per-cell feature pickles are saved

Notes
-----
- NWB files are large (~50–200 MB each). First-run download for all 1920 cells
  will take several hours and ~100–300 GB of disk space.
- Subsequent runs use the local AllenSDK cache (no re-download).
- Spike.fit uses n_jobs=1 per cell; use --workers N to parallelise across cells.
- Cells with <5 detected spikes are skipped (not enough data to fit).
- Sampling rate is read from the sweep and passed to Spike.fit. Most Allen CT
  cells are 200 kHz but a minority differ — this is handled per-cell.
"""

import argparse
import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
ALLEN_CT_HELPER = REPO_ROOT / "AP_empirical_paper_all_analyses/datasets/allen-cell-types/allen_ct_helper_modules"
sys.path.insert(0, str(ALLEN_CT_HELPER))

from config import ALLEN_CT_PICKLE_ROOT
from data_loader import load_cell_metadata, get_all_spiking_sweeps, load_voltage_trace

try:
    from spikewise.patch.fit import Spike
except ImportError:
    sys.path.insert(0, str(REPO_ROOT))
    from spikewise.patch.fit import Spike

FEAT_DIR = os.path.join(ALLEN_CT_PICKLE_ROOT, "features")
os.makedirs(FEAT_DIR, exist_ok=True)

# ── Spike extraction config (matches pvc-6) ───────────────────────────────────
THRESH_MV       = -10    # mV — same as spe-1 and pvc-6
MIN_DIST_MS     = 1.0    # ms minimum between detected spikes
WINDOW_LENGTH   = (5., 5.)
SMOOTH_FRAC     = 0.008
PRE_INFLECTION  = 0.5
MIN_SPIKES      = 5      # skip cells with fewer spikes than this


# =============================================================================
# Per-cell processing
# =============================================================================

def process_cell(specimen_id: int, force: bool = False) -> dict:
    """
    Extract spikewise features for one Allen CT cell, fitting per sweep.

    Saves two pickles:
      {specimen_id}_features.pkl  — spike table (spikewise features + specimen_id + sweep_number + sweep_uid)
      {specimen_id}_sweeps.pkl    — sweep table (one row per spiking sweep + all Allen sweep metadata)
      {specimen_id}_waveforms.pkl — list of per-sweep waveform arrays

    Returns a summary dict: specimen_id, n_spikes, n_sweeps, status, error_msg.
    """
    feat_path  = os.path.join(FEAT_DIR, f"{specimen_id}_features.pkl")
    sweep_path = os.path.join(FEAT_DIR, f"{specimen_id}_sweeps.pkl")
    wav_path   = os.path.join(FEAT_DIR, f"{specimen_id}_waveforms.pkl")

    if not force and os.path.exists(feat_path):
        return dict(specimen_id=specimen_id, status="skipped",
                    n_spikes=None, n_sweeps=None, error_msg=None)

    try:
        spiking_sweeps = get_all_spiking_sweeps(specimen_id)
        if not spiking_sweeps:
            return dict(specimen_id=specimen_id, status="error",
                        n_spikes=0, n_sweeps=0,
                        error_msg="no spiking sweeps found")

        spike_dfs     = []   # one df per sweep → concat → spike table
        sweep_rows    = []   # one dict per sweep → sweep table
        waveform_list = []
        n_sweeps_ok   = 0

        for s in spiking_sweeps:
            sweep_num = s["sweep_number"]
            sweep_uid = f"{specimen_id}_{sweep_num}"
            try:
                voltage, _, _, fs, idx_range = load_voltage_trace(specimen_id, s)
                if idx_range is not None:
                    start, stop = idx_range
                    voltage = voltage[start:stop]

                one_ms = int(fs / 1000)
                idx_spks, _ = find_peaks(voltage, height=THRESH_MV,
                                          distance=int(MIN_DIST_MS * one_ms))
                if len(idx_spks) == 0:
                    continue

                sp = Spike(
                    thresh_amp        = THRESH_MV,
                    window_length     = WINDOW_LENGTH,
                    smooth_frac       = SMOOTH_FRAC,
                    pre_inflection_ms = PRE_INFLECTION,
                )
                sp.fit(list(voltage), fs, n_jobs=1)
                sp.filter_features()

                df = sp.df_features.copy()
                if df.empty:
                    continue

                # Spike table: only IDs — join to sweep/cell tables for metadata
                df["specimen_id"]  = specimen_id
                df["sweep_number"] = sweep_num
                df["sweep_uid"]    = sweep_uid

                spike_dfs.append(df)
                waveform_list.append(
                    sp.waveforms if hasattr(sp, "waveforms") else None
                )

                # Sweep table: all Allen SDK sweep metadata + IDs
                sweep_row = dict(s)          # stimulus_name, amplitude, num_spikes, QC fields, …
                sweep_row["specimen_id"] = specimen_id
                sweep_row["sweep_uid"]   = sweep_uid
                sweep_rows.append(sweep_row)

                n_sweeps_ok += 1

            except Exception:
                continue

        if not spike_dfs:
            return dict(specimen_id=specimen_id, status="error",
                        n_spikes=0, n_sweeps=n_sweeps_ok,
                        error_msg="no sweep produced valid spike features")

        df_spikes = pd.concat(spike_dfs, ignore_index=True)
        df_sweeps = pd.DataFrame(sweep_rows)

        if len(df_spikes) < MIN_SPIKES:
            return dict(specimen_id=specimen_id, status="error",
                        n_spikes=len(df_spikes), n_sweeps=n_sweeps_ok,
                        error_msg=f"too few spikes total ({len(df_spikes)} < {MIN_SPIKES})")

        with open(feat_path,  "wb") as f:
            pickle.dump(df_spikes, f)
        with open(sweep_path, "wb") as f:
            pickle.dump(df_sweeps, f)
        with open(wav_path,   "wb") as f:
            pickle.dump(waveform_list, f)

        return dict(specimen_id=specimen_id, status="ok",
                    n_spikes=len(df_spikes), n_sweeps=n_sweeps_ok,
                    error_msg=None)

    except Exception as e:
        return dict(specimen_id=specimen_id, status="error",
                    n_spikes=None, n_sweeps=None,
                    error_msg=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# =============================================================================
# Population merge
# =============================================================================

def build_population_df(cells_df: pd.DataFrame) -> tuple:
    """
    Concatenate all per-cell pickles into population-level tables.

    Saves three files:
      allen_ct_population_spikes.pkl  — spike table (spikewise features + specimen_id + sweep_uid)
      allen_ct_sweep_table.pkl        — sweep table (one row per spiking sweep, all Allen metadata)
      allen_ct_cells_df.pkl           — cell table (written by load_data notebook, not here)

    Returns (df_spikes, df_sweeps).
    """
    spike_parts = []
    sweep_parts = []

    for sid in cells_df["id"]:
        sid = int(sid)
        feat_path  = os.path.join(FEAT_DIR, f"{sid}_features.pkl")
        sweep_path = os.path.join(FEAT_DIR, f"{sid}_sweeps.pkl")
        for path, dest in [(feat_path, spike_parts), (sweep_path, sweep_parts)]:
            if os.path.exists(path):
                try:
                    dest.append(pickle.load(open(path, "rb")))
                except Exception as e:
                    print(f"  Warning: could not load {path}: {e}")

    if not spike_parts:
        print("No per-cell spike pickles found — run batch first.")
        return pd.DataFrame(), pd.DataFrame()

    df_spikes = pd.concat(spike_parts, ignore_index=True)
    df_sweeps = pd.concat(sweep_parts, ignore_index=True) if sweep_parts else pd.DataFrame()

    spikes_path = os.path.join(ALLEN_CT_PICKLE_ROOT, "allen_ct_population_spikes.pkl")
    sweeps_path = os.path.join(ALLEN_CT_PICKLE_ROOT, "allen_ct_sweep_table.pkl")

    with open(spikes_path, "wb") as f:
        pickle.dump(df_spikes, f)
    with open(sweeps_path, "wb") as f:
        pickle.dump(df_sweeps, f)

    n_cells = df_spikes["specimen_id"].nunique()
    n_sweeps = df_spikes["sweep_uid"].nunique() if "sweep_uid" in df_spikes.columns else "?"
    print(f"Spike table  → {spikes_path}  ({len(df_spikes)} spikes, {n_sweeps} sweeps, {n_cells} cells)")
    print(f"Sweep table  → {sweeps_path}  ({len(df_sweeps)} sweeps)")
    print()
    print("Join pattern:")
    print("  spikes.merge(sweep_table, on='sweep_uid')")
    print("        .merge(cells_df.rename(columns={'id':'specimen_id'}), on='specimen_id')")

    return df_spikes, df_sweeps


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-cells", type=int, default=None,
                   help="Only process the first N cells (useful for testing)")
    p.add_argument("--cells", type=int, nargs="+", default=None,
                   help="Specific specimen IDs to process")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if pickle already exists")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (default 1). Each worker uses n_jobs=1 internally.")
    p.add_argument("--merge-only", action="store_true",
                   help="Skip cell processing; just rebuild population pickle from existing per-cell files")
    return p.parse_args()


def main():
    args = parse_args()

    cells_df = load_cell_metadata(species="Mus musculus")

    if args.cells:
        cells_df = cells_df[cells_df["id"].isin(args.cells)].reset_index(drop=True)
    elif args.n_cells:
        cells_df = cells_df.iloc[:args.n_cells].reset_index(drop=True)

    if args.merge_only:
        build_population_df(cells_df)
        return

    specimen_ids = cells_df["id"].astype(int).tolist()
    print(f"\nProcessing {len(specimen_ids)} cells  |  workers={args.workers}  |  force={args.force}")
    print(f"Output dir: {FEAT_DIR}\n")

    results = []

    if args.workers == 1:
        for sid in tqdm(specimen_ids, desc="cells"):
            r = process_cell(sid, force=args.force)
            results.append(r)
            if r["status"] == "error":
                tqdm.write(f"  [WARN] {sid}: {r['error_msg']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_cell, sid, args.force): sid
                    for sid in specimen_ids}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="cells"):
                r = fut.result()
                results.append(r)
                if r["status"] == "error":
                    tqdm.write(f"  [WARN] {r['specimen_id']}: {r['error_msg']}")

    # Summary
    df_r = pd.DataFrame(results)
    n_ok   = (df_r.status == "ok").sum()
    n_skip = (df_r.status == "skipped").sum()
    n_err  = (df_r.status == "error").sum()
    print(f"\n{'='*60}")
    print(f"Done.  ok={n_ok}  skipped={n_skip}  error={n_err}")
    if n_err:
        print("\nFailed cells:")
        for _, row in df_r[df_r.status == "error"].iterrows():
            print(f"  {row.specimen_id}: {row.error_msg}")

    # Build population pickle from all per-cell files
    print("\nBuilding population DataFrame...")
    build_population_df(cells_df)

    # Save run summary
    summary_path = os.path.join(ALLEN_CT_PICKLE_ROOT, "batch_run_summary.csv")
    df_r.to_csv(summary_path, index=False)
    print(f"Run summary → {summary_path}")


if __name__ == "__main__":
    main()
