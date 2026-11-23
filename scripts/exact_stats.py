#!/usr/bin/env python
"""
Exact quantiles in a LeRobot dataset's meta/stats.json.

LeRobot aggregates the per-episode quantiles of a dataset by their frame-weighted mean (datasets/compute_stats.py,
`aggregate_stats`), which is not a quantile of the data: for a dimension that is constant within each episode but
differs between episodes (the gripper of the insertion tasks), the averaged q01..q99 range collapses by orders of
magnitude, and quantile normalization divides by it. This replaces q01/q10/q50/q90/q99 of the vector features with the
quantiles over all frames (mean/std/min/max aggregate exactly and are only checked).

    python scripts/exact_stats.py <dataset root> [--keys action observation.state]
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("root", type=Path)
parser.add_argument("--keys", nargs="+", default=["action", "observation.state"])
args = parser.parse_args()

path = args.root / "meta" / "stats.json"
stats = json.loads(path.read_text())
frames = pd.concat([pd.read_parquet(f, columns=args.keys) for f in sorted(glob.glob(str(args.root / "data/*/*.parquet")))])
for key in args.keys:
    x = np.stack(frames[key].values).astype(np.float64)
    gap = max(float(np.max(np.abs(np.asarray(stats[key][s]) - f(x))))
              for s, f in (("mean", lambda a: a.mean(0)), ("min", lambda a: a.min(0)), ("max", lambda a: a.max(0))))
    old = np.asarray(stats[key]["q99"]) - np.asarray(stats[key]["q01"])
    for name, q in QUANTILES.items():
        if name in stats[key]:
            stats[key][name] = np.quantile(x, q, axis=0).tolist()
    new = np.asarray(stats[key]["q99"]) - np.asarray(stats[key]["q01"])
    print(f"[exact_stats] {key}: {len(x)} frames, mean/min/max off by {gap:.1e}; q99-q01 smallest dim "
          f"{old.min():.2e} -> {new.min():.2e}")
shutil.copy(path, path.with_name("stats.episode_avg_quantiles.json"))
path.write_text(json.dumps(stats, indent=4))
