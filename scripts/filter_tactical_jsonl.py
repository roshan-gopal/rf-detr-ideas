# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Filter per-frame tactical JSONL for tracker / PickAndRollGraphBuilder.

Reads newline-delimited JSON records (one frame per line) with a
``detections`` array of ``{xyxy, class_id, score, ...}`` and:

* Keeps only **ball** and **player** classes (configurable).
* Maps **player-in-possession** (default ``class_id == 4``) to **player**
  (default ``class_id == 3``) so ``PickAndRollGraphBuilder(ball_class_id=0)``
  sees a single player id for all on-court players.
* Optionally runs **class-aware NMS** (torchvision ``batched_nms``) to drop
  duplicate boxes (e.g. same person as both ``player`` and
  ``player-in-possession``).

Example::

    uv run python scripts/filter_tactical_jsonl.py \\
        raw_frames.jsonl -o filtered_frames.jsonl --nms --iou 0.5

Then build ``sv.Detections`` from each line's ``detections`` and run
``PlayerTracker`` in frame order.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Iterator, TextIO

import numpy as np
import torch
import torchvision.ops

# Defaults aligned with common NBA RF-DETR exports discussed in the project.
DEFAULT_BALL_ID = 0
DEFAULT_PLAYER_ID = 3
DEFAULT_POSSESSION_ID = 4


def filter_detection_dict(
    det: dict[str, Any],
    *,
    ball_id: int,
    player_id: int,
    possession_id: int,
    min_score: float,
) -> dict[str, Any] | None:
    """Return a copy of one detection dict if kept, else ``None``.

    Args:
        det: Raw detection with at least ``class_id``, ``xyxy``, ``score``.
        ball_id: Class id for the ball.
        player_id: Canonical player class id after remapping.
        possession_id: Treated as player; remapped to ``player_id``.

    Returns:
        Filtered detection dict, or ``None`` if dropped.
    """
    if float(det["score"]) < min_score:
        return None

    cid = int(det["class_id"])
    if cid == possession_id:
        cid = player_id
    elif cid == ball_id:
        pass
    elif cid == player_id:
        pass
    else:
        return None

    out = dict(det)
    out["class_id"] = cid
    if "class_name" in out:
        if int(det["class_id"]) == possession_id:
            out["class_name"] = "player"
    return out


def apply_batched_nms(
    xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return indices to keep after torchvision batched NMS.

    Args:
        xyxy: ``(N, 4)`` in pixel ``xyxy`` form.
        scores: ``(N,)`` confidence.
        class_ids: ``(N,)`` integer labels (NMS is not applied across classes).
        iou_threshold: IoU threshold for suppression.

    Returns:
        1-D int64 array of indices into the original arrays.
    """
    if len(xyxy) == 0:
        return np.array([], dtype=np.int64)
    boxes = torch.from_numpy(xyxy.astype(np.float32))
    sc = torch.from_numpy(scores.astype(np.float32))
    cat = torch.from_numpy(class_ids.astype(np.int64))
    keep = torchvision.ops.batched_nms(boxes, sc, cat, iou_threshold)
    return keep.numpy().astype(np.int64)


def filter_frame_record(
    record: dict[str, Any],
    *,
    ball_id: int,
    player_id: int,
    possession_id: int,
    min_score: float,
    nms: bool,
    iou_threshold: float,
) -> dict[str, Any]:
    """Return a new frame record with filtered ``detections`` list."""
    dets_in = record.get("detections", [])
    kept: list[dict[str, Any]] = []
    for d in dets_in:
        fd = filter_detection_dict(
            d,
            ball_id=ball_id,
            player_id=player_id,
            possession_id=possession_id,
            min_score=min_score,
        )
        if fd is not None:
            kept.append(fd)

    if nms and len(kept) > 0:
        xyxy = np.array([x["xyxy"] for x in kept], dtype=np.float32)
        scores = np.array([float(x["score"]) for x in kept], dtype=np.float32)
        cids = np.array([int(x["class_id"]) for x in kept], dtype=np.int64)
        idx = apply_batched_nms(xyxy, scores, cids, iou_threshold)
        kept = [kept[i] for i in idx]

    out = dict(record)
    out["detections"] = kept
    return out


def iter_jsonl_lines(stream: TextIO) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from lines of ``stream``."""
    for line in stream:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Input JSONL path, or '-' for stdin (default: -).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output JSONL path, or '-' for stdout (default: -).",
    )
    parser.add_argument("--ball-id", type=int, default=DEFAULT_BALL_ID, help="Ball class_id to keep.")
    parser.add_argument(
        "--player-id",
        type=int,
        default=DEFAULT_PLAYER_ID,
        help="Canonical player class_id after filtering / remap.",
    )
    parser.add_argument(
        "--possession-id",
        type=int,
        default=DEFAULT_POSSESSION_ID,
        help="Class id remapped to player-id (e.g. player-in-possession).",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Apply class-aware NMS to remaining boxes.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS (default: 0.5).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Drop detections with score below this (default: 0.0).",
    )
    args = parser.parse_args()

    in_stream: TextIO
    out_stream: TextIO
    if args.input == "-":
        in_stream = sys.stdin
    else:
        in_stream = open(args.input, encoding="utf-8")

    if args.output == "-":
        out_stream = sys.stdout
    else:
        out_stream = open(args.output, "w", encoding="utf-8")

    try:
        for record in iter_jsonl_lines(in_stream):
            filtered = filter_frame_record(
                record,
                ball_id=args.ball_id,
                player_id=args.player_id,
                possession_id=args.possession_id,
                min_score=args.min_score,
                nms=args.nms,
                iou_threshold=args.iou,
            )
            out_stream.write(json.dumps(filtered, separators=(",", ":")) + "\n")
    finally:
        if args.input != "-":
            in_stream.close()
        if args.output != "-":
            out_stream.close()


if __name__ == "__main__":
    main()
