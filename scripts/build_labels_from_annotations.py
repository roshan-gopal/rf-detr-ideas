# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Build or refresh ``labels.csv`` from an annotations JSONL.

Writes one row per frame in the annotations file.  Existing labels are copied
when ``(clip_id, frame_index)`` matches ``--merge-labels``; otherwise ``label``
and ``notes`` are left blank for manual annotation.

Example::

    PYTHONPATH=src python scripts/build_labels_from_annotations.py \\
        raw_data_4/annotations.jsonl \\
        -o raw_data_4/labels.csv \\
        --merge-labels raw_data_3/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_existing_labels(path: Path) -> dict[tuple[str, int], tuple[str, str]]:
    """Return ``{(clip_id, frame_index): (label, notes)}`` from a labels CSV."""
    out: dict[tuple[str, int], tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(
            (line for line in f if not line.lstrip().startswith("#")),
            skipinitialspace=True,
        )
        if reader.fieldnames is None:
            raise ValueError(f"Labels CSV {path} has no header row.")
        for row in reader:
            cid = (row.get("clip_id") or "").strip()
            fidx_raw = (row.get("frame_index") or "").strip()
            if not cid or not fidx_raw:
                continue
            try:
                fidx = int(fidx_raw)
            except ValueError:
                continue
            out[(cid, fidx)] = (
                (row.get("label") or "").strip(),
                (row.get("notes") or "").strip(),
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "annotations",
        type=Path,
        help="Input annotations JSONL (e.g. raw_data_4/annotations.jsonl).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output labels CSV path.",
    )
    parser.add_argument(
        "--merge-labels",
        type=Path,
        default=None,
        help="Optional existing labels CSV; matching rows keep label/notes.",
    )
    args = parser.parse_args()

    merged: dict[tuple[str, int], tuple[str, str]] = {}
    if args.merge_labels is not None:
        merged = load_existing_labels(args.merge_labels)
        print(f"Loaded {len(merged)} existing label row(s) from {args.merge_labels}", file=sys.stderr)

    rows: list[dict[str, str]] = []
    for line in args.annotations.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cid = str(rec.get("clip_id", ""))
        fidx = int(rec.get("frame_index", 0))
        ts = rec.get("timestamp_s", 0.0)
        label, notes = merged.get((cid, fidx), ("", ""))
        rows.append(
            {
                "clip_id": cid,
                "frame_index": str(fidx),
                "timestamp_s": f"{float(ts):.2f}",
                "label": label,
                "notes": notes,
            }
        )

    rows.sort(key=lambda r: (r["clip_id"], int(r["frame_index"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["clip_id", "frame_index", "timestamp_s", "label", "notes"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_labeled = sum(1 for r in rows if r["label"] != "")
    clips = {r["clip_id"] for r in rows}
    print(
        f"Wrote {len(rows)} row(s) for {len(clips)} clip(s) -> {args.output} "
        f"({n_labeled} with non-empty label)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
