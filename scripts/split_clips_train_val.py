# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Split a graph JSONL by clip_id into train and val sets.

All frames from one clip go entirely into train OR val — never split across
both.  The split is deterministic given a fixed ``--seed``.

Example::

    python scripts/split_clips_train_val.py \\
        raw_data/graph_all.jsonl \\
        --train raw_data/graph_train.jsonl \\
        --val   raw_data/graph_val.jsonl \\
        --val-fraction 0.2 \\
        --seed 42

After splitting, fit norm stats on the train file only::

    python scripts/compute_graph_feature_stats.py \\
        raw_data/graph_train.jsonl \\
        -o raw_data/graph_feature_norm.json \\
        --only-valid
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl",
        type=Path,
        help="Graph JSONL produced by run_tactical_graph_from_jsonl.py --save-graphs.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output path for training split.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output path for validation split.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        metavar="FRAC",
        help="Fraction of clips to assign to val (default: 0.2 = 20%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        print("--val-fraction must be between 0 and 1 (exclusive).", file=sys.stderr)
        sys.exit(2)

    # Load all rows and group by clip_id
    clip_rows: dict[str, list[str]] = defaultdict(list)
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        clip_id = str(rec.get("clip_id", ""))
        clip_rows[clip_id].append(line)

    clip_ids = sorted(clip_rows.keys())
    n_clips = len(clip_ids)
    if n_clips == 0:
        print("No clips found in input.", file=sys.stderr)
        sys.exit(1)

    n_val = max(1, round(n_clips * args.val_fraction))
    n_train = n_clips - n_val

    if n_train < 1:
        print(
            f"Only {n_clips} clip(s) — not enough for a train/val split at "
            f"val_fraction={args.val_fraction:.2f}.  Reduce --val-fraction or add more clips.",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = random.Random(args.seed)
    shuffled = clip_ids[:]
    rng.shuffle(shuffled)
    val_clips = set(shuffled[:n_val])
    train_clips = set(shuffled[n_val:])

    for path in (args.train, args.val):
        path.parent.mkdir(parents=True, exist_ok=True)

    train_frames = val_frames = 0
    with args.train.open("w", encoding="utf-8") as tf, \
         args.val.open("w", encoding="utf-8") as vf:
        for cid in clip_ids:
            rows = clip_rows[cid]
            if cid in val_clips:
                vf.write("\n".join(rows) + "\n")
                val_frames += len(rows)
            else:
                tf.write("\n".join(rows) + "\n")
                train_frames += len(rows)

    print(f"Clips total : {n_clips}")
    print(f"Train clips : {n_train}  ({train_frames} frames)  → {args.train}")
    print(f"Val   clips : {n_val}   ({val_frames} frames)  → {args.val}")
    print(f"Val clip IDs: {sorted(val_clips)}")
    print(f"\nNext step — fit norm stats on train only:")
    print(
        f"  python scripts/compute_graph_feature_stats.py {args.train} "
        f"-o raw_data/graph_feature_norm.json --only-valid"
    )


if __name__ == "__main__":
    main()
