"""Compare corrected and legacy thresh_ms spike detection behavior.

Usage
-----
python compare_thresh_ms_behavior.py --npy path/to/signal.npy --fs 50000 --thresh-amp 10 --thresh-ms 1.0
"""

import argparse
import numpy as np

from spikeparam.patch.window import find_spike_times, peak_distance_to_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", required=True, help="Path to a 1D NumPy .npy signal.")
    parser.add_argument("--fs", required=True, type=float, help="Sampling rate in Hz.")
    parser.add_argument("--thresh-amp", required=True, type=float, help="Peak height threshold.")
    parser.add_argument("--thresh-ms", default=1.0, type=float, help="Minimum peak distance in ms.")
    parser.add_argument(
        "--preview",
        default=25,
        type=int,
        help="How many detected peak indices to preview from each mode.",
    )
    args = parser.parse_args()

    sig = np.load(args.npy)
    if sig.ndim != 1:
        raise ValueError("Expected a 1D signal in the provided .npy file.")

    corrected_distance = peak_distance_to_samples(args.thresh_ms, args.fs)
    legacy_distance = max(1, int(args.thresh_ms * 1000))

    corrected_inds, _ = find_spike_times(sig, args.thresh_amp, corrected_distance)
    legacy_inds, _ = find_spike_times(sig, args.thresh_amp, legacy_distance)

    corrected_set = set(corrected_inds.tolist())
    legacy_set = set(legacy_inds.tolist())

    print(f"signal length: {len(sig)}")
    print(f"fs: {args.fs}")
    print(f"thresh_amp: {args.thresh_amp}")
    print(f"thresh_ms: {args.thresh_ms}")
    print(f"corrected distance samples: {corrected_distance}")
    print(f"legacy distance samples: {legacy_distance}")
    print(f"corrected spike count: {len(corrected_inds)}")
    print(f"legacy spike count: {len(legacy_inds)}")
    print(f"only corrected: {len(corrected_set - legacy_set)}")
    print(f"only legacy: {len(legacy_set - corrected_set)}")
    print(f"corrected preview: {corrected_inds[:args.preview].tolist()}")
    print(f"legacy preview: {legacy_inds[:args.preview].tolist()}")


if __name__ == "__main__":
    main()
