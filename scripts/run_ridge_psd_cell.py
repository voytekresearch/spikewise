#!/usr/bin/env python3
"""
run_ridge_psd_cell.py
---------------------
Runs the single-PSD ridge regression for ONE spe-1 cell.
Skips (loads from cache) if the pickle already exists, unless --force is given.

Called directly by run_spe1_batch.py --ridge-psd-only, but can also be run
standalone:

    python scripts/run_ridge_psd_cell.py --cell 20
    python scripts/run_ridge_psd_cell.py --cell 20 --force
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
HELPER_DIR  = REPO_ROOT / "AP_empirical_paper_all_analyses" / "datasets" / "spe-1" / "spe1_helper_modules"
sys.path.insert(0, str(HELPER_DIR))

import numpy as np
from config import SPE1_PICKLE_ROOT, LFP_FS
from ridge_regression_utils import (
    load_cell_data, load_hpf_lfp_windows,
    build_ridge_matrices_single_psd,
    run_ridge_regression, apply_fdr,
    WAVEFORM_LABELS,
)

RIDGE_DIR    = os.path.join(SPE1_PICKLE_ROOT, "ridge_regression_pickles")
PRE_WIN      = (-0.055, -0.005)
POST_WIN     = ( 0.005,  0.055)
BASELINE_WIN = (-0.20,  -0.10)
ALPHAS       = np.logspace(-3, 3, 100)
N_PERM       = 1000
RNG_SEED     = 42


def run_psd_ridge(cell_num: int, force: bool = False) -> str:
    """
    Returns 'ok' on success, or an error string on failure.
    Prints progress to stdout.
    """
    save_path = os.path.join(RIDGE_DIR, f"c{cell_num}_ridge_results_psd.pkl")

    if os.path.exists(save_path) and not force:
        print(f"  [psd-ridge c{cell_num}] skipped — pickle exists: {os.path.basename(save_path)}")
        return "ok"

    print(f"  [psd-ridge c{cell_num}] loading data …")
    try:
        df_reg, specparam_by_spike, lfp_windows_by_spike = load_cell_data(cell_num)
        hpf_lfp_01 = load_hpf_lfp_windows(cell_num, hpf_cutoff=0.1)[:len(specparam_by_spike)]

        print(f"  [psd-ridge c{cell_num}] building Y_psd ({len(df_reg)} spikes) …")
        (X_waveform, X_log_isi, X_both, waveform_labels,
         Y_psd, target_names_psd, _) = build_ridge_matrices_single_psd(
            df_reg, lfp_windows_by_spike, hpf_lfp_01, LFP_FS,
            pre_win=PRE_WIN, post_win=POST_WIN, baseline_win=BASELINE_WIN,
            psd_seg_len_s=0.25,
        )

        predictor_sets = {
            "Waveform only":      (X_waveform, waveform_labels),
            "Log ISI only":       (X_log_isi,  ["Log ISI"]),
            "Waveform + Log ISI": (X_both,     waveform_labels + ["Log ISI"]),
        }

        print(f"  [psd-ridge c{cell_num}] running ridge (N_PERM={N_PERM}) …")
        results_psd = run_ridge_regression(
            Y_psd, predictor_sets, target_names_psd,
            n_perm=N_PERM, rng_seed=RNG_SEED, alphas=ALPHAS,
            save_path=save_path, force_recompute=True,
        )
        results_psd = apply_fdr(results_psd, target_names_psd, predictor_sets)
        with open(save_path, "wb") as f:
            pickle.dump(results_psd, f)
        print(f"  [psd-ridge c{cell_num}] saved → {os.path.basename(save_path)}")
        return "ok"

    except Exception:
        import traceback
        return traceback.format_exc()


def main():
    parser = argparse.ArgumentParser(description="Run single-PSD ridge for one spe-1 cell")
    parser.add_argument("--cell",  type=int, required=True, metavar="N",
                        help="Cell number (e.g. 20)")
    parser.add_argument("--force", action="store_true",
                        help="Force rerun even if pickle already exists")
    args = parser.parse_args()

    result = run_psd_ridge(args.cell, force=args.force)
    if result != "ok":
        print(f"\nFAILED c{args.cell}:\n{result}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
