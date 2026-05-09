#!/usr/bin/env python3
"""
run_spe1_batch.py
-----------------
Executes spe-1 per-cell notebooks headlessly via papermill, preserving all
per-cell logic, manual clustering thresholds, and plot output.

Two phases:
  Phase 1 — cluster notebooks (all requested cells)
  Phase 2 — LFP analysis notebooks (priority cells, only when --lfp is passed)

Use --lfp-only to skip clustering entirely and run only LFP notebooks.

Parameters (FORCE_CLUSTER, FORCE_LFP, etc.) are injected into each notebook's
tagged parameters cell, overriding its defaults for that run only.

Usage examples
--------------
# Cluster ALL cells (loads from cache where pickles exist):
    python scripts/run_spe1_batch.py

# Cluster all cells, force-redo even if pickles exist:
    python scripts/run_spe1_batch.py --force-cluster

# Cluster all cells + run LFP for priority cells:
    python scripts/run_spe1_batch.py --lfp

# LFP only for priority cells (skip clustering):
    python scripts/run_spe1_batch.py --lfp-only

# LFP only, force-redo even if pickles exist:
    python scripts/run_spe1_batch.py --lfp-only --force-lfp

# Force-redo everything for priority cells:
    python scripts/run_spe1_batch.py --priority --lfp --force-all

# Specific cells, 4 parallel workers:
    python scripts/run_spe1_batch.py --cells 21 24 42 --lfp --workers 4

Notes
-----
- Each cluster notebook runs spike fitting with n_jobs=-1 (all cores).
  With --workers > 1, notebooks run in parallel — each competing for all cores.
  Recommended: --workers 1 for clustering, higher only if RAM/CPU allows.
- Plots are saved inline into the notebook by papermill automatically.
- Manual thresholds and all per-cell config live in the notebooks — they are
  always respected because papermill runs the actual notebook, not a copy.
"""

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool

import papermill as pm

# ── Path setup ───────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
HELPER_DIR = REPO_ROOT / "AP_empirical_paper1" / "datasets" / "spe-1" / "spe1_helper_modules"
sys.path.insert(0, str(HELPER_DIR))

from config import CELL_IDS, PRIORITY_CELLS

NB_DIR      = (REPO_ROOT / "AP_empirical_paper1" / "datasets" / "spe-1"
               / "spe1_patch_LFP_analysis" / "cluster_analyses")
PRIORITY_SET = {f"c{n}" for n in PRIORITY_CELLS}


def _cell_num(cell_id: str) -> int:
    return int(cell_id.lstrip("c"))


def run_cluster_nb(args):
    """Execute one cluster notebook via papermill. Returns (cell_id, 'ok'|traceback)."""
    cell_id, opts = args
    cnum = _cell_num(cell_id)
    nb   = NB_DIR / f"spe-1_c{cnum}_clusters.ipynb"

    if not nb.exists():
        return (cell_id, f"notebook not found: {nb.name}")

    t0 = datetime.now()
    print(f"  [cluster] c{cnum} started {t0:%H:%M:%S}")
    try:
        pm.execute_notebook(
            str(nb), str(nb),
            parameters={"FORCE_CLUSTER": opts["force_cluster"]},
            kernel_name="python3",
            progress_bar=False,
            log_output=opts["workers"] == 1,
        )
        elapsed = (datetime.now() - t0).seconds // 60
        print(f"  [cluster] c{cnum} done ({elapsed} min)")
        return (cell_id, "ok")
    except Exception:
        return (cell_id, traceback.format_exc())


def run_lfp_nb(args):
    """Execute one LFP analysis notebook via papermill. Returns (cell_id, 'ok'|traceback)."""
    cell_id, opts = args
    cnum = _cell_num(cell_id)
    nb   = NB_DIR / f"spe-1_c{cnum}_LFP_analysis.ipynb"

    if not nb.exists():
        return (cell_id, f"no LFP notebook for c{cnum} (only priority cells have one)")

    t0 = datetime.now()
    print(f"  [LFP]     c{cnum} started {t0:%H:%M:%S}")
    try:
        pm.execute_notebook(
            str(nb), str(nb),
            parameters={
                "FORCE_LFP":    opts["force_lfp"],
                "FORCE_SIMPLE": opts["force_lfp"],
                "FORCE_STATS":  opts["force_lfp"],
            },
            kernel_name="python3",
            progress_bar=False,
            log_output=opts["workers"] == 1,
        )
        elapsed = (datetime.now() - t0).seconds // 60
        print(f"  [LFP]     c{cnum} done ({elapsed} min)")
        return (cell_id, "ok")
    except Exception:
        return (cell_id, traceback.format_exc())


def _run_phase(fn, job_args, workers, label):
    print(f"\n{'='*60}")
    print(f"{label} ({len(job_args)} cells, {workers} worker(s))")
    print(f"{'='*60}")
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(fn, job_args)
    else:
        results = [fn(a) for a in job_args]

    ok  = [cid for cid, s in results if s == "ok"]
    err = [(cid, s) for cid, s in results if s != "ok"]
    print(f"\n  Succeeded: {len(ok)}/{len(job_args)}")
    for cid, tb in err:
        print(f"\n  FAILED: {cid}\n{tb}")
    return ok, err


def main():
    parser = argparse.ArgumentParser(
        description="Run spe-1 cluster/LFP notebooks headlessly via papermill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cells",    nargs="+", type=int, metavar="N",
                        help="Cell numbers to run (e.g. 21 24 42)")
    parser.add_argument("--priority", action="store_true",
                        help="Run only PRIORITY_CELLS from config")
    parser.add_argument("--workers",  type=int, default=1,
                        help="Parallel notebooks (default 1 — see Notes about n_jobs)")
    parser.add_argument("--lfp",      dest="run_lfp",      action="store_true",
                        help="Also run LFP analysis notebooks after clustering (priority cells only)")
    parser.add_argument("--lfp-only", dest="lfp_only",     action="store_true",
                        help="Skip clustering; run LFP analysis notebooks only (priority cells)")
    parser.add_argument("--force-cluster", dest="force_cluster", action="store_true",
                        help="Inject FORCE_CLUSTER=True into cluster notebooks")
    parser.add_argument("--force-lfp",     dest="force_lfp",     action="store_true",
                        help="Inject FORCE_LFP/FORCE_SIMPLE/FORCE_STATS=True into LFP notebooks")
    parser.add_argument("--force-all",     dest="force_all",     action="store_true",
                        help="Equivalent to --force-cluster --force-lfp")
    args = parser.parse_args()

    if args.force_all:
        args.force_cluster = args.force_lfp = True

    # --lfp-only implies --lfp
    if args.lfp_only:
        args.run_lfp = True

    # Resolve cell lists
    if args.cells:
        cell_ids = [f"c{n}" for n in args.cells if f"c{n}" in CELL_IDS]
    elif args.priority:
        cell_ids = [f"c{n}" for n in PRIORITY_CELLS if f"c{n}" in CELL_IDS]
    else:
        cell_ids = list(CELL_IDS)

    lfp_ids = (
        [cid for cid in cell_ids if cid in PRIORITY_SET]
        if args.run_lfp else []
    )

    opts = {"force_cluster": args.force_cluster, "force_lfp": args.force_lfp, "workers": args.workers}

    if args.lfp_only:
        print(f"LFP analysis: {len(lfp_ids)} cells  {sorted(lfp_ids)}  (clustering skipped)")
    else:
        print(f"Clustering : {len(cell_ids)} cells")
        print(f"LFP analysis: {len(lfp_ids)} cells  {sorted(lfp_ids)}")
    print(f"force_cluster={args.force_cluster}  force_lfp={args.force_lfp}  workers={args.workers}")

    # Phase 1: cluster notebooks (skipped when --lfp-only)
    if not args.lfp_only:
        cluster_ok, _ = _run_phase(
            run_cluster_nb,
            [(cid, opts) for cid in cell_ids],
            args.workers,
            "Phase 1: Cluster notebooks",
        )
    else:
        cluster_ok = list(cell_ids)  # treat all as ready since clustering already done

    # Phase 2: LFP analysis notebooks
    if lfp_ids:
        lfp_ready = [cid for cid in lfp_ids if cid in cluster_ok]
        _run_phase(
            run_lfp_nb,
            [(cid, opts) for cid in lfp_ready],
            args.workers,
            "Phase 2: LFP analysis notebooks",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
