#!/usr/bin/env python3
"""
run_spe1_batch.py
-----------------
Executes spe-1 per-cell notebooks headlessly via papermill, preserving all
per-cell logic, manual clustering thresholds, and plot output.

Two phases:
  Phase 1 — cluster notebooks (all requested cells)
  Phase 2 — LFP analysis notebooks

LFP notebook variants:
  --lfp        priority cells  (spe-1_c{N}_LFP_analysis.ipynb)
  --lfp-np     non-priority cells  (spe-1_c{N}_np_LFP_analysis.ipynb)

LFP notebooks run exactly as in manual Jupyter execution — FORCE flags are
controlled inside each notebook's parameters cell, not injected by papermill.
Cluster notebooks still support --force-cluster injection.

Usage examples
--------------
# Cluster ALL cells:
    python scripts/run_spe1_batch.py

# Cluster all cells + run priority LFP:
    python scripts/run_spe1_batch.py --lfp

# LFP only — priority cells (skip clustering):
    python scripts/run_spe1_batch.py --lfp-only

# LFP only — non-priority cells:
    python scripts/run_spe1_batch.py --lfp-np-only

# LFP only — both priority and non-priority:
    python scripts/run_spe1_batch.py --lfp-only --lfp-np-only

# Specific cells (np), 2 workers:
    python scripts/run_spe1_batch.py --lfp-np-only --cells 1 2 6 7 --workers 2

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
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool

import papermill as pm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("papermill").setLevel(logging.INFO)

# ── Path setup ───────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
HELPER_DIR = REPO_ROOT / "AP_empirical_paper1" / "datasets" / "spe-1" / "spe1_helper_modules"
sys.path.insert(0, str(HELPER_DIR))

from config import CELL_IDS, PRIORITY_CELLS

NB_DIR      = (REPO_ROOT / "AP_empirical_paper1" / "datasets" / "spe-1"
               / "spe1_patch_LFP_analysis" / "cluster_analyses" / "cell_analyses")
RIDGE_NB_DIR = NB_DIR / "ridge_regression_analyses"
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


def _run_lfp_notebook(nb_path, label, opts):
    """
    Execute an LFP notebook in-place using jupyter nbconvert --execute.
    Produces output identical to manual Jupyter execution and avoids
    papermill's markdown-cell validation bug.
    """
    import subprocess
    t0 = datetime.now()
    print(f"  [{label}] started {t0:%H:%M:%S}")
    try:
        result = subprocess.run(
            [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=-1",
                "--ExecutePreprocessor.kernel_name=spikeparam",
                str(nb_path),
            ],
            stdout=None if opts["workers"] == 1 else subprocess.DEVNULL,
            stderr=None if opts["workers"] == 1 else subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return f"nbconvert failed (exit {result.returncode})"
        elapsed = (datetime.now() - t0).seconds // 60
        print(f"  [{label}] done ({elapsed} min)")
        return "ok"
    except Exception:
        return traceback.format_exc()


def run_lfp_nb(args):
    """Priority LFP notebook (spe-1_c{N}_LFP_analysis.ipynb)."""
    cell_id, opts = args
    cnum = _cell_num(cell_id)
    nb   = NB_DIR / f"spe-1_c{cnum}_LFP_analysis.ipynb"
    if not nb.exists():
        return (cell_id, f"no priority LFP notebook for c{cnum}")
    result = _run_lfp_notebook(nb, f"LFP c{cnum}", opts)
    return (cell_id, result)


def run_lfp_np_nb(args):
    """Non-priority LFP notebook (spe-1_c{N}_np_LFP_analysis.ipynb)."""
    cell_id, opts = args
    cnum = _cell_num(cell_id)
    nb   = NB_DIR / f"spe-1_c{cnum}_np_LFP_analysis.ipynb"
    if not nb.exists():
        return (cell_id, f"no np LFP notebook for c{cnum}")
    result = _run_lfp_notebook(nb, f"LFP-np c{cnum}", opts)
    return (cell_id, result)


def run_ridge_nb(args):
    """Ridge regression notebook (spe-1_c{N}_spk_to_lfp_ridge.ipynb)."""
    cell_id, opts = args
    cnum = _cell_num(cell_id)
    nb   = RIDGE_NB_DIR / f"spe-1_c{cnum}_spk_to_lfp_ridge.ipynb"
    if not nb.exists():
        return (cell_id, f"no ridge notebook for c{cnum}")
    result = _run_lfp_notebook(nb, f"ridge c{cnum}", opts)
    return (cell_id, result)


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
    parser.add_argument("--lfp",         dest="run_lfp",     action="store_true",
                        help="Run priority LFP notebooks (spe-1_c{N}_LFP_analysis.ipynb)")
    parser.add_argument("--lfp-only",    dest="lfp_only",    action="store_true",
                        help="Skip clustering; run priority LFP notebooks only")
    parser.add_argument("--lfp-np",      dest="run_lfp_np",  action="store_true",
                        help="Run non-priority LFP notebooks (spe-1_c{N}_np_LFP_analysis.ipynb)")
    parser.add_argument("--lfp-np-only", dest="lfp_np_only", action="store_true",
                        help="Skip clustering; run non-priority LFP notebooks only")
    parser.add_argument("--ridge",      dest="run_ridge",   action="store_true",
                        help="Run ridge regression notebooks (spe-1_c{N}_spk_to_lfp_ridge.ipynb)")
    parser.add_argument("--ridge-only", dest="ridge_only",  action="store_true",
                        help="Skip clustering/LFP; run ridge notebooks only")
    parser.add_argument("--force-cluster", dest="force_cluster", action="store_true",
                        help="Inject FORCE_CLUSTER=True into cluster notebooks")
    parser.add_argument("--force-all",     dest="force_all",     action="store_true",
                        help="Equivalent to --force-cluster")
    args = parser.parse_args()

    if args.force_all:
        args.force_cluster = True

    # --*-only flags imply their respective run flags and skip clustering
    if args.lfp_only:    args.run_lfp    = True
    if args.lfp_np_only: args.run_lfp_np = True
    if args.ridge_only:  args.run_ridge  = True

    skip_clustering = args.lfp_only or args.lfp_np_only or args.ridge_only

    # Resolve cell lists
    if args.cells:
        cell_ids = [f"c{n}" for n in args.cells if f"c{n}" in CELL_IDS]
    elif args.priority:
        cell_ids = [f"c{n}" for n in PRIORITY_CELLS if f"c{n}" in CELL_IDS]
    else:
        cell_ids = list(CELL_IDS)

    lfp_ids    = [cid for cid in cell_ids if cid in PRIORITY_SET]     if args.run_lfp    else []
    lfp_np_ids = [cid for cid in cell_ids if cid not in PRIORITY_SET] if args.run_lfp_np else []
    ridge_ids  = list(cell_ids)                                        if args.run_ridge  else []

    opts = {"force_cluster": args.force_cluster, "workers": args.workers}
    # Note: force flags are set inside each notebook's parameters cell

    if not skip_clustering:
        print(f"Clustering:      {len(cell_ids)} cells")
    print(f"LFP (priority):  {len(lfp_ids)} cells  {sorted(lfp_ids)}")
    print(f"LFP (np):        {len(lfp_np_ids)} cells")
    print(f"Ridge:           {len(ridge_ids)} cells")
    print(f"force_cluster={args.force_cluster}  workers={args.workers}")
    print("Note: FORCE flags are set inside each notebook's parameters cell.")

    # Phase 1: cluster notebooks
    if not skip_clustering:
        cluster_ok, _ = _run_phase(
            run_cluster_nb,
            [(cid, opts) for cid in cell_ids],
            args.workers,
            "Phase 1: Cluster notebooks",
        )
    else:
        cluster_ok = list(cell_ids)

    # Phase 2: priority LFP notebooks
    if lfp_ids:
        lfp_ready = [cid for cid in lfp_ids if cid in cluster_ok]
        _run_phase(
            run_lfp_nb,
            [(cid, opts) for cid in lfp_ready],
            args.workers,
            "Phase 2: LFP analysis notebooks (priority)",
        )

    # Phase 3: non-priority LFP notebooks
    if lfp_np_ids:
        np_ready = [cid for cid in lfp_np_ids if cid in cluster_ok]
        _run_phase(
            run_lfp_np_nb,
            [(cid, opts) for cid in np_ready],
            args.workers,
            "Phase 3: LFP analysis notebooks (non-priority)",
        )

    # Phase 4: ridge regression notebooks
    if ridge_ids:
        _run_phase(
            run_ridge_nb,
            [(cid, opts) for cid in ridge_ids],
            args.workers,
            "Phase 4: Ridge regression notebooks",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
