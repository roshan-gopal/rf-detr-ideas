# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Per-dimension mean/std normalization for 63-D ``graph_features`` vectors.

Fit **mean** and **standard deviation** on a **training** set of saved exports
(see ``scripts/run_tactical_graph_from_jsonl.py`` with ``--save-graphs``), save
the stats to JSON, then apply ``(x - mean) / (std + epsilon)`` before the
temporal model. Use the **same** stats at validation and inference.
"""

from __future__ import annotations

__all__ = [
    "GraphFeatureNormalization",
    "compute_mean_std",
    "standardize",
]

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rfdetr.graph import GRAPH_FEATURE_DIM


@dataclass
class GraphFeatureNormalization:
    """Mean and std for each of the ``GRAPH_FEATURE_DIM`` dimensions."""

    mean: np.ndarray  # shape (GRAPH_FEATURE_DIM,), float32
    std: np.ndarray  # shape (GRAPH_FEATURE_DIM,), float32
    count: int  # number of vectors used to fit

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "count": self.count,
            "graph_feature_dim": GRAPH_FEATURE_DIM,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> GraphFeatureNormalization:
        """Load from ``to_json_dict`` output or compatible JSON."""
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.asarray(data["std"], dtype=np.float32)
        count = int(data["count"])
        if mean.shape != (GRAPH_FEATURE_DIM,) or std.shape != (GRAPH_FEATURE_DIM,):
            raise ValueError(
                f"expected mean/std shape ({GRAPH_FEATURE_DIM},), got {mean.shape} / {std.shape}"
            )
        return cls(mean=mean, std=std, count=count)

    def save(self, path: str | Path) -> None:
        """Write stats to a JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GraphFeatureNormalization:
        """Load stats from :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json_dict(data)


def compute_mean_std(features: np.ndarray, *, ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension mean and standard deviation.

    Args:
        features: Array of shape ``(N, GRAPH_FEATURE_DIM)`` with ``N >= 1``.
        ddof: Passed to ``numpy.std``. Use ``1`` (sample std) for training-set
            statistics; if ``N == 1``, ``ddof`` is treated as ``0``.

    Returns:
        ``(mean, std)`` each of shape ``(GRAPH_FEATURE_DIM,)``, dtype float32.
    """
    if features.ndim != 2 or features.shape[1] != GRAPH_FEATURE_DIM:
        raise ValueError(
            f"expected features of shape (N, {GRAPH_FEATURE_DIM}), got {features.shape}"
        )
    n = features.shape[0]
    if n == 0:
        raise ValueError("features must have at least one row")
    eff_ddof = ddof if n > 1 else 0
    mean = np.mean(features, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(features, axis=0, dtype=np.float64, ddof=eff_ddof).astype(np.float32)
    return mean, std


def standardize(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Return ``(x - mean) / (std + epsilon)`` with broadcasting.

    Args:
        x: Shape ``(..., GRAPH_FEATURE_DIM)`` (e.g. one row ``(63,)`` or batch
            ``(T, 63)`` or ``(B, T, 63)``).
        mean: Shape ``(GRAPH_FEATURE_DIM,)``.
        std: Shape ``(GRAPH_FEATURE_DIM,)``.
        epsilon: Added to std to avoid division by zero.

    Returns:
        Same shape as ``x``, float32.
    """
    if mean.shape != (GRAPH_FEATURE_DIM,) or std.shape != (GRAPH_FEATURE_DIM,):
        raise ValueError(f"mean/std must have shape ({GRAPH_FEATURE_DIM},)")
    m = mean.astype(np.float32)
    s = (std.astype(np.float32) + np.float32(epsilon))
    return ((x.astype(np.float32) - m) / s).astype(np.float32)
