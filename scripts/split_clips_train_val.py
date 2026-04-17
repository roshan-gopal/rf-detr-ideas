# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Split graph JSONL (and optionally frame labels CSV) by clip_id into train/val.

Every frame in a clip stays together — never split across train and val.

**Graph split** (required): reads ``graph_all.jsonl``, writes ``--train`` and
``--val`` JSONL paths.

**Assignment** (pick one):

- ``--val-fraction`` + ``--seed`` — random partition of clips (default).
- ``--val-clips`` — comma-separated ``clip_id`` values forced into val; all
  other clips go to train (overrides ``--val-fraction``).

**Labels split** (optional): pass ``--labels`` plus ``--labels-train`` and
``--labels-val``; each row is copied to the file matching its ``clip_id``.
Rows whose ``clip_id`` is missing from the graph JSONL are skipped with a
warning.

Examples::

    # Random 20%% val clips
    python scripts/split_clips_train_val.py graph_all.jsonl \\
        --train graph_train.jsonl --val graph_val.jsonl \\
        --val-fraction 0.2 --seed 42

    # Exactly one clip in val (by name)
    python scripts/split_clips_train_val.py graph_all.jsonl \\
        --train graph_train.jsonl --val graph_val.jsonl \\
        --val-clips Dame_no_screen

    # Graph + labels together
    python scripts/split_clips_train_val.py graph_all.jsonl \\
        --train graph_train.jsonl --val graph_val.jsonl \\
        --val-clips Haliburton_Test \\
        --labels labels.csv \\
        --labels-train labels_train.csv \\
        --labels-val labels_val.csv

After splitting, fit norm stats on train only::

    python scripts/compute_graph_feature_stats.py graph_train.jsonl \\
        -o graph_feature_norm.json --only-valid
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _parse_val_clips(raw: str) -> set[str]:
    """Return non-empty clip ids from a comma-separated string."""
    return {s.strip() for s in raw.split(",") if s.strip()}


def _split_labels_csv(
    labels_path: Path,
    train_out: Path,
    val_out: Path,
    train_clips: set[str],
    val_clips: set[str],
    graph_clip_ids: set[str],
) -> None:
    """Write label rows to train or val CSV by ``clip_id``."""
    lines_in = labels_path.read_text(encoding="utf-8").splitlines()
    non_comment = [ln for ln in lines_in if not ln.lstrip().startswith("#")]
    if not non_comment:
        print(f"No data rows in {labels_path}.", file=sys.stderr)
        sys.exit(1)
    header_line = non_comment[0]
    reader = csv.DictReader(non_comment, skipinitialspace=True)
    if reader.fieldnames is None:
        print(f"No header in {labels_path}.", file=sys.stderr)
        sys.exit(1)
    fieldnames = list(reader.fieldnames)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in reader:
        cid = (row.get("clip_id") or "").strip()
        if not cid:
            skipped += 1
            continue
        if cid not in graph_clip_ids:
            print(f"Warning: label row clip_id={cid!r} not in graph JSONL — skipped.", file=sys.stderr)
            skipped += 1
            continue
        if cid in val_clips:
            val_rows.append(row)
        elif cid in train_clips:
            train_rows.append(row)
        else:
            print(f"Warning: clip_id={cid!r} not in train or val set — skipped.", file=sys.stderr)
            skipped += 1

    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    with train_out.open("w", encoding="utf-8", newline="") as tf:
        w = csv.DictWriter(tf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(train_rows)
    with val_out.open("w", encoding="utf-8", newline="") as vf:
        w = csv.DictWriter(vf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(val_rows)

    print(f"Labels train : {len(train_rows)} rows → {train_out}")
    print(f"Labels val   : {len(val_rows)} rows → {val_out}")
    if skipped:
        print(f"Labels skipped rows: {skipped}", file=sys.stderr)


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
        help="Fraction of clips to assign to val when --val-clips is not used (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when using --val-fraction (default: 42).",
    )
    parser.add_argument(
        "--val-clips",
        type=str,
        default=None,
        metavar="IDS",
        help="Comma-separated clip_id values to place entirely in val (overrides --val-fraction).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        metavar="PATH",
        help="Frame-level labels CSV (clip_id, frame_index, label, ...).",
    )
    parser.add_argument(
        "--labels-train",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output CSV for label rows whose clip_id is in the train split.",
    )
    parser.add_argument(
        "--labels-val",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output CSV for label rows whose clip_id is in the val split.",
    )
    args = parser.parse_args()

    labels_requested = args.labels is not None or args.labels_train is not None or args.labels_val is not None
    if labels_requested:
        if args.labels is None or args.labels_train is None or args.labels_val is None:
            print("--labels, --labels-train, and --labels-val must be given together.", file=sys.stderr)
            sys.exit(2)

    # Load all graph rows and group by clip_id
    clip_rows: dict[str, list[str]] = defaultdict(list)
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        clip_id = str(rec.get("clip_id", ""))
        clip_rows[clip_id].append(line)

    clip_ids = sorted(clip_rows.keys())
    graph_clip_ids = set(clip_ids)
    n_clips = len(clip_ids)
    if n_clips == 0:
        print("No clips found in input.", file=sys.stderr)
        sys.exit(1)

    if args.val_clips is not None:
        val_clips = _parse_val_clips(args.val_clips)
        unknown = val_clips - graph_clip_ids
        if unknown:
            print(f"Unknown --val-clips not in graph JSONL: {sorted(unknown)}", file=sys.stderr)
            sys.exit(1)
        if not val_clips:
            print("--val-clips must name at least one clip_id.", file=sys.stderr)
            sys.exit(2)
        train_clips = graph_clip_ids - val_clips
        if not train_clips:
            print("All clips would be in val — leave at least one clip for train.", file=sys.stderr)
            sys.exit(1)
        n_val = len(val_clips)
        n_train = len(train_clips)
    else:
        if not (0.0 < args.val_fraction < 1.0):
            print("--val-fraction must be between 0 and 1 (exclusive).", file=sys.stderr)
            sys.exit(2)
        n_val = max(1, round(n_clips * args.val_fraction))
        n_train = n_clips - n_val
        if n_train < 1:
            print(
                f"Only {n_clips} clip(s) — not enough for a train/val split at "
                f"val_fraction={args.val_fraction:.2f}.  Use --val-clips or reduce val_fraction.",
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
    with args.train.open("w", encoding="utf-8") as tf, args.val.open("w", encoding="utf-8") as vf:
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
    print(f"Train clip IDs: {sorted(train_clips)}")
    print(f"Val clip IDs:   {sorted(val_clips)}")

    if args.labels is not None:
        _split_labels_csv(
            args.labels,
            args.labels_train,
            args.labels_val,
            train_clips,
            val_clips,
            graph_clip_ids,
        )

    print("\nNext step — fit norm stats on train only:")
    print(
        f"  python scripts/compute_graph_feature_stats.py {args.train} "
        f"-o raw_data/graph_feature_norm.json --only-valid"
    )


if __name__ == "__main__":
    main()
