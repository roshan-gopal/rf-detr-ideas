
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Run PlayerTracker + PickAndRollGraphBuilder on filtered JSONL.

Expects each line to match ``scripts/filter_tactical_jsonl.py`` output:
``detections`` with ``xyxy``, ``class_id``, ``score``, plus ``fps``,
``width``, ``height``.

Use ``--save-graphs PATH`` to write one JSON object per line for training:
``graph_features`` (length ``GRAPH_FEATURE_DIM``, :func:`rfdetr.graph.flatten_graph_frame`), plus
``node_features`` / ``edge_features`` / ``graph_valid`` for inspection.

Use ``--show-features N`` to print the flattened vector for the first ``N`` frames
(so you can eyeball the exact training input without opening the file).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv

from rfdetr.graph import GRAPH_FEATURE_DIM, PickAndRollGraphBuilder, flatten_graph_frame
from rfdetr.tracking import PlayerTracker


def record_to_detections(record: dict[str, Any]) -> sv.Detections:
    """Build ``sv.Detections`` from one frame record."""
    dets = record.get("detections", [])
    if not dets:
        return sv.Detections.empty()
    xyxy = np.array([d["xyxy"] for d in dets], dtype=np.float32)
    class_id = np.array([int(d["class_id"]) for d in dets], dtype=np.int32)
    confidence = np.array([float(d["score"]) for d in dets], dtype=np.float32)
    return sv.Detections(xyxy=xyxy, class_id=class_id, confidence=confidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl",
        type=Path,
        help="Path to filtered JSONL (e.g. raw_data/annotations_filtered.jsonl).",
    )
    parser.add_argument(
        "--save-graphs",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Write per-frame JSONL with graph_features ({GRAPH_FEATURE_DIM}) and tensor breakdown.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-frame stdout lines (still prints final summary).",
    )
    parser.add_argument(
        "--show-features",
        type=int,
        default=0,
        metavar="N",
        help=f"Print graph_features ({GRAPH_FEATURE_DIM} floats) for the first N frames.",
    )
    args = parser.parse_args()

    lines = args.jsonl.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    records.sort(key=lambda r: (r.get("clip_id", ""), r.get("frame_index", 0)))

    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    fps = float(records[0]["fps"])
    width = int(records[0]["width"])
    height = int(records[0]["height"])
    tracker = PlayerTracker(fps=fps)
    builder = PickAndRollGraphBuilder(
        image_width=width,
        image_height=height,
        ball_class_id=0,
    )

    out_fp = None
    if args.save_graphs is not None:
        args.save_graphs.parent.mkdir(parents=True, exist_ok=True)
        out_fp = args.save_graphs.open("w", encoding="utf-8")

    n_valid = 0
    n_players_ge_3 = 0
    n_shown_features = 0
    prev_clip: str | None = None
    try:
        for rec in records:
            clip = str(rec.get("clip_id", ""))
            if prev_clip is not None and clip != prev_clip:
                tracker.reset()
                builder.reset()
            prev_clip = clip
            dets = record_to_detections(rec)
            if dets.class_id is not None:
                n_players_ge_3 += int((dets.class_id == 3).sum() >= 3)
            tracked = tracker.update(dets)
            graph = builder.build(tracked)
            if graph.valid:
                n_valid += 1

            gf = flatten_graph_frame(graph)
            row: dict[str, Any] = {
                "clip_id": rec.get("clip_id"),
                "frame_index": rec.get("frame_index"),
                "timestamp_s": rec.get("timestamp_s"),
                "fps": rec.get("fps"),
                "width": rec.get("width"),
                "height": rec.get("height"),
                "graph_valid": graph.valid,
                "graph_features": gf.astype(float).tolist(),
                "node_features": graph.node_features.astype(float).tolist(),
                "edge_features": graph.edge_features.astype(float).tolist(),
            }
            if out_fp is not None:
                out_fp.write(json.dumps(row, separators=(",", ":")) + "\n")

            if args.show_features > 0 and n_shown_features < args.show_features:
                feat_str = ", ".join(f"{float(x):.6f}" for x in gf.tolist())
                print(
                    f"frame_index={rec.get('frame_index')} graph_valid={graph.valid} "
                    f"graph_features=[{feat_str}]"
                )
                n_shown_features += 1

            if not args.quiet:
                print(
                    f"frame_index={rec.get('frame_index')} "
                    f"n_det={len(tracked)} valid_graph={graph.valid}"
                )
    finally:
        if out_fp is not None:
            out_fp.close()

    print(
        f"\nSummary: {len(records)} frames, "
        f"{n_players_ge_3} frames with ≥3 player boxes (pre-track), "
        f"{n_valid} frames with valid 3-body graph (post-track)."
    )
    if args.save_graphs is not None:
        print(f"Wrote training rows to {args.save_graphs}", file=sys.stderr)


if __name__ == "__main__":
    main()
