# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Frame-level training loop for PickAndRollTemporalClassifier.

Loads per-frame graph features from ``--graphs`` JSONL (produced by
``scripts/run_tactical_graph_from_jsonl.py --save-graphs``), groups frames by
``clip_id``, and trains ``PickAndRollTemporalClassifier`` against **per-frame**
binary labels from ``--labels`` CSV.

Label format (CSV, one row per labeled frame)::

    clip_id,frame_index,label[,notes]

Unlabeled frames (those not in the CSV) are excluded from the loss — the model
still sees them and uses them for context in the Transformer, but they do not
contribute a gradient.  This lets you label only the frames you are confident
about (e.g. the exact window where the screen is set) without needing to label
every single frame.

Smoke-test mode (``--smoke-test``) skips the labels file and assigns label=1 to
the middle third of every clip — useful to verify shapes, gradients, and the
masked-loss logic without real labels.

Example (smoke test)::

    PYTHONPATH=src python scripts/train_pick_and_roll.py \\
        --graphs raw_data/graph_train.jsonl \\
        --smoke-test

Example (real labels, 5 epochs)::

    PYTHONPATH=src python scripts/train_pick_and_roll.py \\
        --graphs raw_data/graph_train.jsonl \\
        --labels raw_data/labels.csv \\
        --norm-stats raw_data/graph_feature_norm.json \\
        --epochs 5

Structured logs (for plotting or analysis)::

    PYTHONPATH=src python scripts/train_pick_and_roll.py \\
        --graphs raw_data/graph_train.jsonl \\
        --labels raw_data/labels.csv \\
        --log-jsonl raw_data/logs/train_metrics.jsonl \\
        --log-csv raw_data/logs/train_metrics.csv

JSONL has one JSON object per line: ``run_start``, ``train_step``, and
``epoch_summary`` records.  CSV uses the same events in tabular form (empty
cells where a field does not apply).  Plot epoch curves from CSV with e.g.
``pandas``: filter ``event == "epoch_summary"`` and column ``avg_frame_loss``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
import torch.nn as nn

from rfdetr.graph import GRAPH_FEATURE_DIM
from rfdetr.graph_normalization import GraphFeatureNormalization, standardize
from rfdetr.temporal import PickAndRollTemporalClassifier, PickAndRollTemporalEncoder


# ── helpers ───────────────────────────────────────────────────────────────


def load_frame_labels(path: Path) -> dict[tuple[str, int], float]:
    """Return ``{(clip_id, frame_index): label}`` from a CSV file.

    Args:
        path: CSV file with columns ``clip_id``, ``frame_index``, ``label``
            (and optional ``notes``).  Lines starting with ``#`` are skipped.

    Returns:
        Mapping from ``(clip_id, frame_index)`` to binary label float.
    """
    labels: dict[tuple[str, int], float] = {}
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
            lbl_raw = (row.get("label") or "").strip()
            if not cid or not fidx_raw or not lbl_raw:
                continue
            try:
                labels[(cid, int(fidx_raw))] = float(lbl_raw)
            except ValueError as exc:
                print(f"Skipping bad label row {row}: {exc}", file=sys.stderr)
    return labels


def load_graph_features(
    path: Path,
) -> dict[str, list[tuple[int, np.ndarray, bool]]]:
    """Return ``{clip_id: [(frame_index, gf, graph_valid), ...]}`` sorted by frame.

    Args:
        path: JSONL file produced by ``run_tactical_graph_from_jsonl.py --save-graphs``.

    Returns:
        Per-clip list of ``(frame_index, feature_vector, graph_valid)`` tuples,
        sorted by ``frame_index``.
    """
    clips: dict[str, list[tuple[int, np.ndarray, bool]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec: dict[str, Any] = json.loads(line)
        cid = str(rec.get("clip_id", ""))
        fidx = int(rec.get("frame_index", 0))
        valid = bool(rec.get("graph_valid", False))
        gf_raw = rec.get("graph_features")
        if gf_raw is None:
            continue
        gf = np.asarray(gf_raw, dtype=np.float32)
        if gf.shape != (GRAPH_FEATURE_DIM,):
            continue
        clips[cid].append((fidx, gf, valid))
    for cid in clips:
        clips[cid].sort(key=lambda t: t[0])
    return dict(clips)


def build_sequence(
    frames: list[tuple[int, np.ndarray, bool]],
    frame_labels: dict[tuple[str, int], float],
    clip_id: str,
    norm: GraphFeatureNormalization | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build tensors for one clip.

    Args:
        frames: List of ``(frame_index, feature_vector, graph_valid)`` tuples.
        frame_labels: Mapping from ``(clip_id, frame_index)`` to binary label.
        clip_id: Clip identifier (used to look up labels).
        norm: Optional normalization stats.
        device: Target torch device.

    Returns:
        Tuple of:
        - ``x``: ``(1, T, GRAPH_FEATURE_DIM)`` feature tensor.
        - ``padding_mask``: ``(1, T)`` bool mask — ``True`` = invalid frame.
        - ``targets``: ``(1, T)`` float label tensor (``NaN`` for unlabeled frames).
        - ``label_mask``: ``(1, T)`` bool mask — ``True`` = frame has a label.
    """
    rows: list[torch.Tensor] = []
    ignore: list[bool] = []
    target_list: list[float] = []
    has_label: list[bool] = []

    for fidx, gf, valid in frames:
        feat = standardize(gf, norm.mean, norm.std) if norm is not None else gf
        rows.append(torch.from_numpy(feat).to(device))
        ignore.append(not valid)
        key = (clip_id, fidx)
        if key in frame_labels:
            target_list.append(frame_labels[key])
            has_label.append(True)
        else:
            target_list.append(float("nan"))
            has_label.append(False)

    x = torch.stack(rows, dim=0).unsqueeze(0)                              # (1, T, D)
    padding_mask = torch.tensor([ignore], dtype=torch.bool, device=device)  # (1, T)
    targets = torch.tensor([target_list], dtype=torch.float32, device=device)  # (1, T)
    label_mask = torch.tensor([has_label], dtype=torch.bool, device=device)    # (1, T)
    return x, padding_mask, targets, label_mask


def _utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _write_train_log_jsonl(fp: TextIO, record: dict[str, Any]) -> None:
    """Append one JSON object (one line) to a JSONL log file."""
    fp.write(json.dumps(record, separators=(",", ":")) + "\n")
    fp.flush()


def _write_train_log_csv_row(
    writer: csv.DictWriter,
    fp: TextIO,
    row: dict[str, str],
) -> None:
    """Write one CSV row and flush so progress survives crashes."""
    writer.writerow(row)
    fp.flush()


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphs",
        type=Path,
        required=True,
        help="JSONL from run_tactical_graph_from_jsonl.py --save-graphs.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="CSV with columns clip_id,frame_index,label. Required unless --smoke-test.",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="JSON from compute_graph_feature_stats.py (optional but recommended).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Assign label=1 to the middle third of each clip; skip labels file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of passes over the clip set (default: 1).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu).",
    )
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSONL log path (overwritten each run): run_start, train_step, epoch_summary lines.",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="CSV log for spreadsheets / pandas (same events as JSONL; empty cells where N/A).",
    )
    args = parser.parse_args()

    if not args.smoke_test and args.labels is None:
        print("Provide --labels or pass --smoke-test.", file=sys.stderr)
        sys.exit(2)

    device = torch.device(args.device)

    norm: GraphFeatureNormalization | None = None
    if args.norm_stats is not None:
        norm = GraphFeatureNormalization.load(args.norm_stats)
        print(f"Loaded norm stats from {args.norm_stats} (N={norm.count})")

    clips = load_graph_features(args.graphs)
    if not clips:
        print("No clips loaded from graphs JSONL.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(clips)} clip(s): {list(clips.keys())}")

    frame_labels: dict[tuple[str, int], float]
    if args.smoke_test:
        frame_labels = {}
        for cid, frames in clips.items():
            n = len(frames)
            lo, hi = n // 3, 2 * n // 3
            for i, (fidx, _gf, _valid) in enumerate(frames):
                frame_labels[(cid, fidx)] = 1.0 if lo <= i < hi else 0.0
        total_labeled = sum(1 for (c, _) in frame_labels if c in clips)
        print(f"Smoke-test mode: {total_labeled} frame labels generated (middle-third=1).")
    else:
        frame_labels = load_frame_labels(args.labels)
        print(f"Loaded {len(frame_labels)} frame label(s).")
        missing_clips = [c for c in clips if not any(c == k[0] for k in frame_labels)]
        if missing_clips:
            print(
                f"Warning: no labels at all for clips {missing_clips} — they will be skipped.",
                file=sys.stderr,
            )

    # Keep only clips that have at least one labeled frame
    labeled_clips = [
        (cid, frames)
        for cid, frames in clips.items()
        if any((cid, f[0]) in frame_labels for f in frames)
    ]
    if not labeled_clips:
        print("No labeled frames found to train on.", file=sys.stderr)
        sys.exit(1)
    print(f"Training on {len(labeled_clips)} clip(s) with at least one labeled frame.")

    model = PickAndRollTemporalClassifier(
        encoder=PickAndRollTemporalEncoder(
            embed_dim=GRAPH_FEATURE_DIM,
            num_heads=3,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        ),
        frame_level=True,
    ).to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    log_jsonl_fp: TextIO | None = None
    log_csv_fp: TextIO | None = None
    log_csv_writer: csv.DictWriter | None = None
    csv_fieldnames = [
        "event",
        "time_utc",
        "epoch",
        "epochs_total",
        "global_step",
        "clip_id",
        "loss",
        "n_labeled",
        "avg_frame_loss",
        "labeled_frames_epoch",
    ]

    if args.log_jsonl is not None:
        args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        log_jsonl_fp = args.log_jsonl.open("w", encoding="utf-8")
        _write_train_log_jsonl(
            log_jsonl_fp,
            {
                "kind": "run_start",
                "time_utc": _utc_now_iso(),
                "graphs": str(args.graphs.resolve()),
                "epochs": args.epochs,
                "lr": args.lr,
                "device": args.device,
                "smoke_test": args.smoke_test,
                "norm_stats": str(args.norm_stats.resolve()) if args.norm_stats else None,
            },
        )
        print(f"Writing JSONL metrics to {args.log_jsonl}", file=sys.stderr)

    if args.log_csv is not None:
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        log_csv_fp = args.log_csv.open("w", encoding="utf-8", newline="")
        log_csv_writer = csv.DictWriter(log_csv_fp, fieldnames=csv_fieldnames, extrasaction="ignore")
        log_csv_writer.writeheader()
        log_csv_fp.flush()
        _write_train_log_csv_row(
            log_csv_writer,
            log_csv_fp,
            {
                "event": "run_start",
                "time_utc": _utc_now_iso(),
                "epoch": "",
                "epochs_total": str(args.epochs),
                "global_step": "",
                "clip_id": "",
                "loss": "",
                "n_labeled": "",
                "avg_frame_loss": "",
                "labeled_frames_epoch": "",
            },
        )
        print(f"Writing CSV metrics to {args.log_csv}", file=sys.stderr)

    global_step = 0

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            epoch_labeled_frames = 0

            for cid, frames in labeled_clips:
                x, padding_mask, targets, label_mask = build_sequence(
                    frames, frame_labels, cid, norm, device
                )
                logits = model(x, src_key_padding_mask=padding_mask)  # (1, T)

                # Apply loss only on labeled, valid frames
                active = label_mask & ~padding_mask                    # (1, T)
                n_active = active.sum().item()
                if n_active == 0:
                    continue

                loss = criterion(logits[active], targets[active]) / n_active
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1
                epoch_loss += loss.item() * n_active
                epoch_labeled_frames += int(n_active)

                probs = torch.sigmoid(logits[0])                       # (T,)
                labeled_frame_indices = [
                    (fi, int(label_mask[0, t].item()), float(probs[t].item()), float(targets[0, t].item()))
                    for t, (fi, _gf, _valid) in enumerate(frames)
                    if label_mask[0, t].item()
                ]
                prob_summary = " ".join(
                    f"f{fi}:lbl={int(lbl)},p={p:.2f}"
                    for fi, _has, p, lbl in labeled_frame_indices
                )
                print(
                    f"  epoch={epoch} clip={cid!r} n_labeled={int(n_active)} "
                    f"loss={loss.item():.4f} [{prob_summary}]"
                )

                if log_jsonl_fp is not None:
                    _write_train_log_jsonl(
                        log_jsonl_fp,
                        {
                            "kind": "train_step",
                            "time_utc": _utc_now_iso(),
                            "epoch": epoch,
                            "epochs_total": args.epochs,
                            "global_step": global_step,
                            "clip_id": cid,
                            "loss": loss.item(),
                            "n_labeled": int(n_active),
                        },
                    )

                if log_csv_writer is not None and log_csv_fp is not None:
                    _write_train_log_csv_row(
                        log_csv_writer,
                        log_csv_fp,
                        {
                            "event": "train_step",
                            "time_utc": _utc_now_iso(),
                            "epoch": str(epoch),
                            "epochs_total": str(args.epochs),
                            "global_step": str(global_step),
                            "clip_id": cid,
                            "loss": f"{loss.item():.8f}",
                            "n_labeled": str(int(n_active)),
                            "avg_frame_loss": "",
                            "labeled_frames_epoch": "",
                        },
                    )

            if epoch_labeled_frames > 0:
                avg = epoch_loss / epoch_labeled_frames
            else:
                avg = float("nan")
            print(
                f"Epoch {epoch}/{args.epochs}  avg_frame_loss={avg:.4f}  labeled_frames={epoch_labeled_frames}"
            )

            if log_jsonl_fp is not None:
                _write_train_log_jsonl(
                    log_jsonl_fp,
                    {
                        "kind": "epoch_summary",
                        "time_utc": _utc_now_iso(),
                        "epoch": epoch,
                        "epochs_total": args.epochs,
                        "global_step_end": global_step,
                        "avg_frame_loss": avg,
                        "labeled_frames_epoch": epoch_labeled_frames,
                    },
                )

            if log_csv_writer is not None and log_csv_fp is not None:
                avg_str = f"{avg:.8f}" if math.isfinite(avg) else ""
                _write_train_log_csv_row(
                    log_csv_writer,
                    log_csv_fp,
                    {
                        "event": "epoch_summary",
                        "time_utc": _utc_now_iso(),
                        "epoch": str(epoch),
                        "epochs_total": str(args.epochs),
                        "global_step": str(global_step),
                        "clip_id": "",
                        "loss": "",
                        "n_labeled": "",
                        "avg_frame_loss": avg_str,
                        "labeled_frames_epoch": str(epoch_labeled_frames),
                    },
                )
    finally:
        if log_jsonl_fp is not None:
            log_jsonl_fp.close()
        if log_csv_fp is not None:
            log_csv_fp.close()

    print("\nDone. Loss should decrease across epochs if training is working.")
    if args.smoke_test:
        print("(smoke-test run — no real labels; results are not meaningful)")


if __name__ == "__main__":
    main()
