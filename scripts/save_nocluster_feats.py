#!/usr/bin/env python3
"""
save_nocluster_feats.py
-----------------------
Creates cluster_df.pkl for cells that have spike_fit pickles but no waveform
clusters (c17, c18, c43).  Saves df_features (waveform params + log_isi +
spk_times_ms + spk_id) without any *_cluster columns so the ridge regression
pipeline can include these cells.

Usage:
    python scripts/save_nocluster_feats.py
"""

import sys
import os
import pickle
from pathlib import Path
import numpy as np

REPO_ROOT  = Path(__file__).resolve().parent.parent
HELPER_DIR = REPO_ROOT / "AP_empirical_paper1" / "datasets" / "spe-1" / "spe1_helper_modules"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HELPER_DIR))

from config import SPE1_PICKLE_ROOT, DICT_PATCH_FS

NO_CLUSTER_CELLS = [17, 18, 43]

for cnum in NO_CLUSTER_CELLS:
    print(f"\n--- c{cnum} ---")

    fit_path     = os.path.join(SPE1_PICKLE_ROOT, "spike_fit_pickles", f"c{cnum}_spike_fit.pkl")
    cluster_path = os.path.join(SPE1_PICKLE_ROOT, "cluster_pickles",   f"c{cnum}_cluster_df.pkl")

    if not os.path.exists(fit_path):
        print(f"  SKIP — spike_fit pickle not found: {fit_path}")
        continue

    sp = pickle.load(open(fit_path, "rb"))

    df = sp.df_features.copy()

    # Spike times in ms (spike_inds / (fs / 1000))
    patch_fs = DICT_PATCH_FS[cnum]
    df["spk_times_ms"] = sp.spike_inds / (patch_fs / 1000.0)
    df["spk_id"]       = range(len(df))

    # Drop spk_times_idx if it somehow exists
    if "spk_times_idx" in df.columns:
        df = df.drop(columns=["spk_times_idx"])

    with open(cluster_path, "wb") as f:
        pickle.dump(df, f)

    print(f"  Saved {len(df)} spikes → {os.path.basename(cluster_path)}")
    print(f"  Columns: {list(df.columns)}")

print("\nDone.")
