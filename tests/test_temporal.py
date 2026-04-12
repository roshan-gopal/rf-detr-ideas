# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for pick-and-roll temporal encoder and graph-sequence helpers."""

from __future__ import annotations

import numpy as np
import torch

from rfdetr.graph import EDGE_DIM, GRAPH_FEATURE_DIM, GraphFrame, NODE_DIM
from rfdetr.temporal import (
    PickAndRollTemporalClassifier,
    PickAndRollTemporalEncoder,
    SinusoidalPositionEncoding,
    encode_graph_sequence,
)


def test_sinusoidal_position_encoding_shape() -> None:
    """PE preserves (B, T, D) shape."""
    d_model = GRAPH_FEATURE_DIM
    pe = SinusoidalPositionEncoding(d_model=d_model, max_len=64, dropout=0.0)
    x = torch.zeros(2, 10, d_model)
    y = pe(x)
    assert y.shape == (2, 10, d_model)


def test_temporal_encoder_output_shape() -> None:
    """Encoder maps (B, T, D) to (B, T, D) — full contextualised sequence."""
    enc = PickAndRollTemporalEncoder(
        embed_dim=GRAPH_FEATURE_DIM,
        num_heads=3,
        num_layers=1,
        dim_feedforward=128,
    )
    x = torch.randn(3, 15, GRAPH_FEATURE_DIM)
    z = enc(x)
    assert z.shape == (3, 15, GRAPH_FEATURE_DIM)


def test_temporal_encoder_padding_mask_runs() -> None:
    """Forward with padding mask completes without NaNs (regression guard)."""
    enc = PickAndRollTemporalEncoder(
        embed_dim=GRAPH_FEATURE_DIM,
        num_heads=3,
        num_layers=2,
        dim_feedforward=128,
    )
    x = torch.randn(1, 8, GRAPH_FEATURE_DIM)
    mask = torch.tensor([[False, False, True, True, False, False, False, False]], dtype=torch.bool)
    z = enc(x, src_key_padding_mask=mask)
    assert z.shape == (1, 8, GRAPH_FEATURE_DIM)
    assert torch.isfinite(z).all()


def test_temporal_classifier_frame_level_logit_shape() -> None:
    """Frame-level classifier outputs one logit per frame (B, T)."""
    clf = PickAndRollTemporalClassifier(
        embed_dim=GRAPH_FEATURE_DIM,
        num_heads=3,
        num_layers=1,
        dim_feedforward=128,
        frame_level=True,
    )
    x = torch.randn(4, 8, GRAPH_FEATURE_DIM)
    logits = clf(x)
    assert logits.shape == (4, 8)


def test_temporal_classifier_clip_level_logit_shape() -> None:
    """Clip-level classifier outputs one logit per clip (B,)."""
    clf = PickAndRollTemporalClassifier(
        embed_dim=GRAPH_FEATURE_DIM,
        num_heads=3,
        num_layers=1,
        dim_feedforward=128,
        frame_level=False,
    )
    x = torch.randn(4, 8, GRAPH_FEATURE_DIM)
    logits = clf(x)
    assert logits.shape == (4,)


def test_encode_graph_sequence_shape() -> None:
    """encode_graph_sequence returns (1, T, GRAPH_FEATURE_DIM) and padding mask."""
    graphs: list[GraphFrame] = []
    for _ in range(5):
        graphs.append(
            GraphFrame(
                node_features=np.random.randn(3, NODE_DIM).astype(np.float32),
                edge_features=np.random.randn(6, EDGE_DIM).astype(np.float32),
                valid=True,
            )
        )
    x, pad = encode_graph_sequence(graphs)
    assert x.shape == (1, 5, GRAPH_FEATURE_DIM)
    assert pad.shape == (1, 5)
    assert pad.dtype == torch.bool
    assert not pad.any()


def test_encode_graph_sequence_marks_invalid_frames() -> None:
    """Invalid GraphFrame positions are True in src_key_padding_mask."""
    zero_nodes = np.zeros((3, NODE_DIM), dtype=np.float32)
    zero_edges = np.zeros((6, EDGE_DIM), dtype=np.float32)
    graphs = [
        GraphFrame(node_features=zero_nodes, edge_features=zero_edges, valid=True),
        GraphFrame(node_features=zero_nodes, edge_features=zero_edges, valid=False),
        GraphFrame(node_features=zero_nodes, edge_features=zero_edges, valid=True),
    ]
    _x, pad = encode_graph_sequence(graphs)
    assert pad.tolist() == [[False, True, False]]
