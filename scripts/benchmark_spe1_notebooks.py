"""Benchmark representative spe-1 notebook pipelines on real data."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SPE1_HELPER_DIR = REPO_ROOT / "AP_empirical_paper_all_analyses/datasets/spe-1/spe1_helper_modules"
SHARED_HELPER_DIR = REPO_ROOT / "AP_empirical_paper_all_analyses/datasets/shared_helper_modules"

for helper_dir in (SPE1_HELPER_DIR, SHARED_HELPER_DIR):
    helper_path = str(helper_dir)
    if helper_path not in sys.path:
        sys.path.insert(0, helper_path)

from config import DICT_PATCH_FS, DICT_SPK_THRESH
from spikewise.patch.fit import Spike
from spk_feat_cluster_analysis import (
    cluster_multimodal_features,
    plot_full_cluster_report,
    extract_lfp_windows,
    build_lfp_groups_from_clusters,
    compute_simple_lfp_by_spike,
    run_master_LFP_spk_analysis,
    load_chunked_specparam_results,
)


DATA_ROOT = Path(
    "/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/"
    "Neuropixel Paired Recordings/Recordings"
)
PICKLE_ROOT = Path("/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/spe1_pickles")
LFP_FS = 3500

CELL_CONFIGS = {
    21: {
        "patch_path": DATA_ROOT / "filt_patch_recordings/c21_patch.npy",
        "lfp_path": DATA_ROOT / "filt_lfp_recordings/c21_lfp.npy",
        "thresh_amp": 4.0,
        "cluster_kwargs": {
            "max_k": 3,
            "suffix": "_cluster",
            "plot_each": False,
            "manual_thresholds": {
                "peak_amp": (9, 14),
                "peak_sharpness": 1.5,
                "peak_width": (0.35, 0.5),
            },
        },
        "specparam_chunk_dir": PICKLE_ROOT / "multitaper_pickles/c21",
        "cluster_pickle_path": PICKLE_ROOT / "cluster_pickles/c21_df_clusters.pkl",
        "run_lfp_stage": True,
    },
    34: {
        "patch_path": DATA_ROOT / "filt_patch_recordings/c34_patch.npy",
        "lfp_path": DATA_ROOT / "filt_lfp_recordings/c34_lfp.npy",
        "thresh_amp": None,
        "cluster_kwargs": {
            "max_k": 2,
            "suffix": "_cluster",
            "plot_each": False,
            "manual_thresholds": {},
        },
        "specparam_chunk_dir": PICKLE_ROOT / "multitaper_pickles/c34",
        "cluster_pickle_path": PICKLE_ROOT / "cluster_pickles/c34_df_clusters.pkl",
        "run_lfp_stage": False,
    },
}


def _time_stage(name: str, func, timings: list[tuple[str, float]]):
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    timings.append((name, elapsed))
    print(f"{name}: {elapsed:.2f}s")
    return result


def benchmark_cell(cell_num: int) -> dict[str, object]:
    if cell_num not in CELL_CONFIGS:
        raise ValueError(f"Unsupported cell {cell_num}")

    cfg = CELL_CONFIGS[cell_num]
    patch_fs = DICT_PATCH_FS[cell_num]
    thresh_amp = cfg["thresh_amp"] if cfg["thresh_amp"] is not None else DICT_SPK_THRESH[cell_num]
    timings: list[tuple[str, float]] = []

    patch_filt = _time_stage("load_patch", lambda: np.load(cfg["patch_path"]), timings)
    lfp_filt = _time_stage("load_lfp", lambda: np.load(cfg["lfp_path"]), timings)

    patch_times = _time_stage(
        "build_patch_times",
        lambda: np.arange(0, patch_filt.shape[0]) / (patch_fs / 1000.0),
        timings,
    )
    lfp_times = _time_stage(
        "build_lfp_times",
        lambda: np.arange(0, lfp_filt.shape[0]) / (LFP_FS / 1000.0),
        timings,
    )

    sp = Spike(thresh_amp=thresh_amp, window_length=(5.0, 5.0), smooth_frac=0.01)
    _time_stage(
        "spike_fit",
        lambda: sp.fit(patch_filt, patch_fs, n_jobs=-1, progress=tqdm, flip_signal=False),
        timings,
    )
    _time_stage("filter_features", lambda: sp.filter_features(), timings)

    df_features = _time_stage("copy_features", lambda: sp.df_features.copy(), timings)
    df_features["spk_times_idx"] = sp.spike_inds
    df_features["spk_times_ms"] = patch_times[sp.spike_inds]
    df_features["spk_id"] = range(len(df_features))

    cluster_input = df_features.drop(["spk_times_idx"], axis=1) if cell_num == 21 else df_features
    df_features_clust, rep = _time_stage(
        "cluster_multimodal_features",
        lambda: cluster_multimodal_features(cluster_input, **cfg["cluster_kwargs"]),
        timings,
    )
    plt.close("all")

    cluster_cols = [col for col in df_features_clust.columns if col.endswith("_cluster")]
    all_reports = []

    def _run_cluster_reports():
        for i, col in enumerate(cluster_cols):
            all_reports.append(plot_full_cluster_report(df_features_clust, sp, col, cluster_group_id=i))
        return pd.concat(all_reports, ignore_index=True) if all_reports else pd.DataFrame()

    master_report_df = _time_stage("plot_full_cluster_report_loop", _run_cluster_reports, timings)
    plt.close("all")

    results: dict[str, object] = {
        "cell_num": cell_num,
        "timings": timings,
        "n_spikes": len(sp.spike_inds),
        "n_cluster_cols": len(cluster_cols),
        "cluster_report_rows": len(master_report_df),
        "clustered_features_shape": tuple(df_features_clust.shape),
        "cluster_pickle_path": str(cfg["cluster_pickle_path"]),
        "cluster_diagnostics_keys": sorted(rep.keys()),
    }

    if cfg["run_lfp_stage"]:
        all_spike_extractor = _time_stage(
            "extract_lfp_windows",
            lambda: extract_lfp_windows(
                spk_df=df_features_clust,
                lfp_times_ms=lfp_times,
                lfp_signal=lfp_filt,
                pre_s=1.0,
                post_s=1.5,
                condition=None,
            ),
            timings,
        )
        specparam_by_spike = _time_stage(
            "load_chunked_specparam_results",
            lambda: load_chunked_specparam_results(
                save_dir=str(cfg["specparam_chunk_dir"]),
                prefix=f"c{cell_num}_specparam",
            ),
            timings,
        )
        groups = _time_stage(
            "build_lfp_groups_from_clusters",
            lambda: build_lfp_groups_from_clusters(df_features_clust, all_spike_extractor),
            timings,
        )
        lfp_windows_by_spike = _time_stage(
            "compute_simple_lfp_by_spike",
            lambda: compute_simple_lfp_by_spike(
                windows_all=all_spike_extractor["windows"],
                times_rel_list=all_spike_extractor["times_rel_ms"],
                fs=LFP_FS,
                inner_window_s=0.5,
                step_s=0.025,
                freq_range=(4, 90),
            ),
            timings,
        )
        features_list = [
            {"feature": "exponent", "label": "Aperiodic Exponent"},
            {"feature": "offset", "label": "Aperiodic Offset"},
            {"feature": "band", "band": "theta", "label": "Theta AUC"},
            {"feature": "band", "band": "beta", "label": "Beta AUC"},
            {"feature": "band", "band": "gamma", "label": "Gamma AUC"},
            {"feature": "r_squared", "label": "r-squared"},
            {"feature": "lfp_mean", "label": "LFP Mean Amplitude"},
            {"feature": "lfp_std", "label": "LFP Std"},
            {"feature": "lfp_exponent", "label": "LFP Spectral Exponent"},
        ]
        master_results = _time_stage(
            "run_master_LFP_spk_analysis",
            lambda: run_master_LFP_spk_analysis(
                cell_id=str(cell_num),
                specparam_by_spike=specparam_by_spike,
                groups=groups,
                features_to_analyze=features_list,
                lfp_windows_by_spike=lfp_windows_by_spike,
                window_width=0.05,
                step_size=0.025,
                p_threshold=0.05,
            ),
            timings,
        )
        results.update(
            {
                "n_lfp_windows": len(all_spike_extractor["windows"]),
                "n_specparam_entries": len(specparam_by_spike),
                "n_group_keys": len(groups),
                "n_master_results": len(master_results) if hasattr(master_results, "__len__") else None,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", type=int, default=[21, 34])
    args = parser.parse_args()

    print(f"Benchmarking cells: {args.cells}")
    for cell_num in args.cells:
        print(f"\n=== CELL {cell_num} ===")
        results = benchmark_cell(cell_num)
        total = sum(elapsed for _, elapsed in results["timings"])
        print(f"total: {total:.2f}s")
        print(f"n_spikes: {results['n_spikes']}")
        print(f"n_cluster_cols: {results['n_cluster_cols']}")
        if "n_lfp_windows" in results:
            print(f"n_lfp_windows: {results['n_lfp_windows']}")
            print(f"n_specparam_entries: {results['n_specparam_entries']}")


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    main()
