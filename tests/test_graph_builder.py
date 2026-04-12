# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for PickAndRollGraphBuilder ball-handler heuristics."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from rfdetr.graph import PickAndRollGraphBuilder


def _frame(
    xyxy: np.ndarray,
    class_id: np.ndarray,
    tracker_id: np.ndarray,
    velocity: np.ndarray | None = None,
    speed: np.ndarray | None = None,
) -> sv.Detections:
    """Build Detections with velocity/speed matching ``PlayerTracker`` output."""
    n = len(xyxy)
    if velocity is None:
        velocity = np.zeros((n, 2), dtype=np.float32)
    if speed is None:
        speed = np.linalg.norm(velocity, axis=1).astype(np.float32)
    confidence = np.ones((n,), dtype=np.float32)
    dets = sv.Detections(
        xyxy=xyxy.astype(np.float32),
        class_id=class_id.astype(np.int32),
        confidence=confidence,
        tracker_id=tracker_id.astype(np.int32),
    )
    dets.data = {"velocity": velocity.astype(np.float32), "speed": speed.astype(np.float32)}
    return dets


def test_reset_clears_ball_handler_memory() -> None:
    """reset() drops last ball-handler track and position."""
    builder = PickAndRollGraphBuilder(
        image_width=1280,
        image_height=720,
        ball_class_id=0,
    )
    ball = np.array([[630.0, 350.0, 650.0, 370.0]], dtype=np.float32)
    players = np.array(
        [
            [600.0, 340.0, 640.0, 380.0],
            [0.0, 0.0, 50.0, 100.0],
            [800.0, 400.0, 840.0, 500.0],
        ],
        dtype=np.float32,
    )
    xyxy = np.vstack([ball, players])
    cid = np.array([0, 3, 3, 3], dtype=np.int32)
    tid = np.array([0, 7, 8, 9], dtype=np.int32)
    builder.build(_frame(xyxy, cid, tid))
    assert builder._last_bh_tracker_id == 7
    assert builder._last_bh_centre is not None
    builder.reset()
    assert builder._last_bh_tracker_id is None
    assert builder._last_bh_centre is None


def test_no_ball_prefers_last_ball_handler_track_not_fastest() -> None:
    """Without ball, BH stays the prior track ID even if another player is faster."""
    builder = PickAndRollGraphBuilder(
        image_width=1280,
        image_height=720,
        ball_class_id=0,
        screen_radius=500.0,
    )
    # Frame 1: ball near player A (tracker 10).
    ball = np.array([[630.0, 350.0, 650.0, 370.0]], dtype=np.float32)
    player_a = np.array([[610.0, 345.0, 635.0, 375.0]], dtype=np.float32)
    player_b = np.array([[100.0, 100.0, 140.0, 200.0]], dtype=np.float32)
    player_c = np.array([[900.0, 400.0, 940.0, 500.0]], dtype=np.float32)
    xyxy1 = np.vstack([ball, player_a, player_b, player_c])
    cid1 = np.array([0, 3, 3, 3], dtype=np.int32)
    tid1 = np.array([0, 10, 11, 12], dtype=np.int32)
    g1 = builder.build(_frame(xyxy1, cid1, tid1))
    assert g1.valid
    bh_node1 = g1.node_features[0]
    # BH should correspond to player A (left side of frame).
    assert bh_node1[0] < 0.55

    # Frame 2: no ball; player B is much faster; same track IDs.
    xyxy2 = np.vstack(
        [
            player_a + np.array([[20.0, 0.0, 20.0, 0.0]]),
            player_b,
            player_c,
        ],
    )
    cid2 = np.array([3, 3, 3], dtype=np.int32)
    tid2 = np.array([10, 11, 12], dtype=np.int32)
    vel2 = np.array([[0.0, 0.0], [800.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    g2 = builder.build(_frame(xyxy2, cid2, tid2, velocity=vel2))
    assert g2.valid
    bh_node2 = g2.node_features[0]
    # Still player with track 10 (shifted right slightly), not the fast player at left.
    assert bh_node2[0] > 0.45
    assert bh_node2[0] < 0.55


def test_cold_start_no_ball_uses_fastest_player() -> None:
    """With no prior BH memory, no ball implies argmax speed among players."""
    builder = PickAndRollGraphBuilder(image_width=1280, image_height=720, ball_class_id=0)
    xyxy = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [100.0, 100.0, 120.0, 140.0],
            [200.0, 200.0, 220.0, 240.0],
        ],
        dtype=np.float32,
    )
    cid = np.array([3, 3, 3], dtype=np.int32)
    tid = np.array([1, 2, 3], dtype=np.int32)
    vel = np.array([[1.0, 0.0], [500.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    g = builder.build(_frame(xyxy, cid, tid, velocity=vel))
    assert g.valid
    # Middle player (index 1) has highest speed → BH node x should match that box center.
    cx_mid = (xyxy[1, 0] + xyxy[1, 2]) / 2.0 / 1280.0
    assert pytest.approx(g.node_features[0, 0], rel=1e-5) == cx_mid
