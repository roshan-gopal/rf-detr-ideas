# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compute mean/std for ``graph_features`` from a saved-graph JSONL (training set).

Vector length must match :data:`rfdetr.graph.GRAPH_FEATURE_DIM` (defined in
``rfdetr.graph`` from the current flattened graph layout — not a fixed magic
number). Re-run this script after changing node/edge features so ``mean`` /
``std`` match your exported JSONL.

Example::

    PYTHONPATH=src python scripts/compute_graph_feature_stats.py \\
        raw_data/graph_train.jsonl -o raw_data/graph_feature_norm.json

Use :class:`rfdetr.graph_normalization.GraphFeatureNormalization` to load the
JSON and :func:`rfdetr.graph_normalization.standardize` on arrays before the
temporal encoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from rfdetr.graph import GRAPH_FEATURE_DIM
from rfdetr.graph_normalization import GraphFeatureNormalization, compute_mean_std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl",
        type=Path,
        help="JSONL from run_tactical_graph_from_jsonl.py --save-graphs (training split).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output JSON path (mean, std, count).",
    )
    parser.add_argument(
        "--only-valid",
        action="store_true",
        help="Use only lines where graph_valid is true (recommended).",
    )
    args = parser.parse_args()

    rows: list[np.ndarray] = []
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if args.only_valid and not rec.get("graph_valid", False):
            continue
        gf = rec.get("graph_features")
        if gf is None:
            print("Skipping line without graph_features.", file=sys.stderr)
            continue
        v = np.asarray(gf, dtype=np.float32)
        if v.shape != (GRAPH_FEATURE_DIM,):
            print(f"Skipping bad shape {v.shape}, expected ({GRAPH_FEATURE_DIM},).", file=sys.stderr)
            continue
        rows.append(v)

    if not rows:
        print("No graph_features collected.", file=sys.stderr)
        sys.exit(1)

    mat = np.stack(rows, axis=0)
    mean, std = compute_mean_std(mat, ddof=1)
    stats = GraphFeatureNormalization(mean=mean, std=std, count=len(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats.save(args.output)
    print(f"Wrote {stats.count} samples, dim={GRAPH_FEATURE_DIM} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
