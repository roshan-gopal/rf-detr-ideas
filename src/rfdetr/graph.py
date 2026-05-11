# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""3-body relational graph builder for pick-and-roll analytics.

Each frame is represented as a directed complete graph over three players:

* **Node 0 — Ball-handler (BH):** player closest to the ball detection,
  or the player who **last** had the ball (same ByteTrack ID), falling back
  to the player nearest that last known position if the ID is missing; only
  on a cold start (no prior ball-handler) is the fastest player used.
* **Node 1 — Screener (S):** slowest player within ``screen_radius`` pixels
  of the ball-handler. This heuristic fires on the player most likely to be
  planting to set a screen.
* **Node 2 — Nearest defender (D):** player closest to the ball-handler
  who is neither the ball-handler nor the screener.

Node features (``NODE_DIM = 11``)::

    [cx_norm, cy_norm, vx_norm, vy_norm, speed_norm, w_norm, h_norm,
     bh_rel_x, bh_rel_y, heading_x, heading_y]

Where:

* ``bh_rel_x / bh_rel_y`` — position relative to the ball-handler
  (zero for the BH node itself); removes absolute court-zone bias.
* ``heading_x / heading_y`` — unit velocity direction ``v / (|v| + ε)``;
  captures movement direction independent of speed magnitude so that a
  play running left vs right produces the same directional pattern.

Edge features (``EDGE_DIM = 9``, one row per directed edge A→B)::

    [dx_norm, dy_norm, dvx_norm, dvy_norm, dist_norm, approach_rate,
     speed_ratio, blocker_score, screen_angle_cos]

Where:

* ``approach_rate = dot(v_A_norm, unit(A→B))`` captures whether A is
  actively moving *toward* B (positive) or away (negative) — the key signal
  for distinguishing a player running past another from one approaching to use
  a screen.
* ``speed_ratio = speed_A / (speed_B + ε)`` captures whether one player is
  stationary relative to the other — a high ratio on the S→BH edge means the
  screener is planted while the ball-handler is in motion.
* ``blocker_score`` — how squarely the screener is positioned on the BH→D
  line; ``1 - clip(perp_dist(S, BH→D) / |BH→D|, 0, 1)``.  Value near 1
  means S is directly in the defender's path (textbook screen).  Identical
  for all six edges since it is a frame-level scalar.
* ``screen_angle_cos`` — cosine of the angle between the BH→S direction and
  BH's own velocity heading.  Near 1 means the ball-handler is running
  directly toward the screen; near −1 means running away.  Identical for all
  six edges.

Example::

    from rfdetr.tracking import PlayerTracker
    from rfdetr.graph import PickAndRollGraphBuilder, flatten_graph_frame

    tracker = PlayerTracker(fps=30.0)
    builder = PickAndRollGraphBuilder(image_width=1280, image_height=720)

    for frame in video_frames:
        detections = model.predict(frame, threshold=0.5)
        tracked = tracker.update(detections)
        graph = builder.build(tracked)
        features_63 = flatten_graph_frame(graph)  # zeros if ``not graph.valid``
"""

from __future__ import annotations

__all__ = [
    "NODE_DIM",
    "EDGE_DIM",
    "GRAPH_FEATURE_DIM",
    "GraphFrame",
    "PickAndRollGraphBuilder",
    "flatten_graph_frame",
]

from dataclasses import dataclass

import numpy as np
import supervision as sv

# Number of features per node and per directed edge — exposed so downstream
# modules can derive their input dimensions without hard-coding magic numbers.
NODE_DIM: int = 11
EDGE_DIM: int = 9
_NUM_NODES: int = 3
_NUM_EDGES: int = 6  # complete directed graph: 3 nodes × 2 directions
# Flattened ``GraphFrame``: ``3 * NODE_DIM + 6 * EDGE_DIM`` (node rows + edge rows).
GRAPH_FEATURE_DIM: int = _NUM_NODES * NODE_DIM + _NUM_EDGES * EDGE_DIM
# Edge order is fixed: (BH→S, S→BH, BH→D, D→BH, S→D, D→S)
_EDGE_PAIRS: list[tuple[int, int]] = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]


@dataclass
class GraphFrame:
    """Node and edge feature arrays for one frame's 3-body graph.

    Attributes:
        node_features: Float32 array of shape ``(3, NODE_DIM)``.  Rows are
            ordered ``[ball-handler, screener, defender]``.
        edge_features: Float32 array of shape ``(6, EDGE_DIM)``.  Rows
            correspond to directed edges in the order
            ``(BH→S, S→BH, BH→D, D→BH, S→D, D→S)``.
        valid: ``False`` when fewer than 3 players were detected; downstream
            modules should skip or zero-pad this frame.
    """

    node_features: np.ndarray
    edge_features: np.ndarray
    valid: bool


def flatten_graph_frame(graph: GraphFrame) -> np.ndarray:
    """Concatenate node and edge features into one float32 vector of length ``GRAPH_FEATURE_DIM``.

    Invalid frames return a zero vector so sequence models can use padding masks.

    Args:
        graph: One frame from :meth:`PickAndRollGraphBuilder.build`.

    Returns:
        Array of shape ``(GRAPH_FEATURE_DIM,)``, dtype ``float32``.
    """
    if not graph.valid:
        return np.zeros((GRAPH_FEATURE_DIM,), dtype=np.float32)
    return np.concatenate(
        [graph.node_features.reshape(-1), graph.edge_features.reshape(-1)]
    ).astype(np.float32, copy=False)


class PickAndRollGraphBuilder:
    """Builds a per-frame 3-body relational graph from tracked detections.

    Args:
        image_width: Source frame width in pixels — used to normalise
            horizontal positions and box widths to ``[0, 1]``.
        image_height: Source frame height in pixels — used to normalise
            vertical positions and box heights to ``[0, 1]``.
        ball_class_id: Class ID that the detector assigns to the ball.
            Detections with this ID are excluded from player role assignment.
        screen_radius: Maximum pixel distance from the ball-handler within
            which a screener candidate is searched.  Players outside this
            radius are only considered if no closer player exists.
        max_speed: Speed in pixels/second used to clip and normalise velocity
            components.  Values above this are clamped to ``±1``.
    """

    def __init__(
        self,
        image_width: int = 1280,
        image_height: int = 720,
        ball_class_id: int = 1,
        screen_radius: float = 200.0,
        max_speed: float = 1000.0,
    ) -> None:
        self.W = float(image_width)
        self.H = float(image_height)
        self.ball_class_id = ball_class_id
        self.screen_radius = screen_radius
        self.max_speed = max_speed
        self._last_bh_tracker_id: int | None = None
        self._last_bh_centre: np.ndarray | None = None

    def reset(self) -> None:
        """Clear ball-handler memory when switching to a new clip or game."""
        self._last_bh_tracker_id = None
        self._last_bh_centre = None

    def _ball_handler_idx_without_ball(
        self,
        *,
        centres: np.ndarray,
        p_speed: np.ndarray,
        tid_player: np.ndarray | None,
    ) -> int:
        """Pick ball-handler among players when no ball box is present."""
        if self._last_bh_tracker_id is not None and tid_player is not None:
            matches = np.nonzero(tid_player == self._last_bh_tracker_id)[0]
            if len(matches) > 0:
                return int(matches[0])
        if self._last_bh_centre is not None:
            return int(np.argmin(np.linalg.norm(centres - self._last_bh_centre, axis=1)))
        return int(np.argmax(p_speed))

    def build(self, tracked: sv.Detections) -> GraphFrame:
        """Build a ``GraphFrame`` from one frame of tracked detections.

        Args:
            tracked: Output of ``PlayerTracker.update()``.  Must have
                ``tracker_id`` set and ``data["velocity"]`` /
                ``data["speed"]`` populated.

        Returns:
            ``GraphFrame`` with ``valid=True`` when at least 3 player
            detections are present, ``valid=False`` otherwise.
        """
        _empty = GraphFrame(
            node_features=np.zeros((_NUM_NODES, NODE_DIM), dtype=np.float32),
            edge_features=np.zeros((_NUM_EDGES, EDGE_DIM), dtype=np.float32),
            valid=False,
        )

        if tracked.class_id is None or len(tracked) == 0:
            return _empty

        is_ball = tracked.class_id == self.ball_class_id
        is_player = ~is_ball

        if is_player.sum() < _NUM_NODES:
            return _empty

        # ── Player geometry ───────────────────────────────────────────────
        p_xyxy = tracked.xyxy[is_player]
        p_vel = tracked.data["velocity"][is_player]   # (N, 2) px/s
        p_speed = tracked.data["speed"][is_player]    # (N,)   px/s

        cx = (p_xyxy[:, 0] + p_xyxy[:, 2]) / 2.0
        cy = (p_xyxy[:, 1] + p_xyxy[:, 3]) / 2.0
        w = p_xyxy[:, 2] - p_xyxy[:, 0]
        h = p_xyxy[:, 3] - p_xyxy[:, 1]
        centres = np.stack([cx, cy], axis=1)   # (N, 2)
        n = len(cx)

        # ── Role assignment ───────────────────────────────────────────────
        # Ball-handler: nearest player to ball; if no ball, last ball-handler
        # track ID, else nearest to last BH centre, else fastest (cold start).
        tid_full = tracked.tracker_id
        tid_player: np.ndarray | None = tid_full[is_player] if tid_full is not None else None

        ball_xyxy = tracked.xyxy[is_ball]
        if len(ball_xyxy) > 0:
            ball_c = np.array(
                [(ball_xyxy[0, 0] + ball_xyxy[0, 2]) / 2.0, (ball_xyxy[0, 1] + ball_xyxy[0, 3]) / 2.0]
            )
            bh_idx = int(np.argmin(np.linalg.norm(centres - ball_c, axis=1)))
        else:
            bh_idx = self._ball_handler_idx_without_ball(
                centres=centres,
                p_speed=p_speed,
                tid_player=tid_player,
            )

        if tid_player is not None:
            self._last_bh_tracker_id = int(tid_player[bh_idx])
        self._last_bh_centre = centres[bh_idx].astype(np.float64, copy=True)

        # Screener: slowest player within screen_radius of ball-handler.
        dist_from_bh = np.linalg.norm(centres - centres[bh_idx], axis=1)
        near = (dist_from_bh < self.screen_radius) & (np.arange(n) != bh_idx)
        pool = np.where(near)[0] if near.any() else np.array([i for i in range(n) if i != bh_idx])
        screener_idx = int(pool[np.argmin(p_speed[pool])])

        # Defender: nearest remaining player to ball-handler.
        exclude = {bh_idx, screener_idx}
        remaining = [i for i in range(n) if i not in exclude]
        if remaining:
            defender_idx = remaining[int(np.argmin(dist_from_bh[remaining]))]
        else:
            defender_idx = screener_idx  # degenerate — only 2 players visible

        roles = [bh_idx, screener_idx, defender_idx]

        # ── Frame-level geometric features (shared across all edges) ──────
        bh_c = centres[bh_idx]
        s_c = centres[screener_idx]
        d_c = centres[defender_idx]

        # Blocker score: how squarely S lies on the BH→D line.
        # perp_dist = |cross(BH→D, BH→S)| / |BH→D|; normalise by |BH→D|.
        bd = d_c - bh_c
        bs = s_c - bh_c
        bd_len = float(np.linalg.norm(bd))
        cross = abs(float(bd[0] * bs[1] - bd[1] * bs[0]))
        perp_dist = cross / (bd_len + 1e-6)
        blocker_score = float(np.clip(1.0 - perp_dist / (bd_len + 1e-6), 0.0, 1.0))

        # Screen angle cos: cosine of angle between BH velocity and BH→S direction.
        bs_len = float(np.linalg.norm(bs))
        bs_unit = bs / (bs_len + 1e-6)
        bh_heading = p_vel[bh_idx] / (float(p_speed[bh_idx]) + 1e-6)
        screen_angle_cos = float(np.clip(np.dot(bh_heading, bs_unit), -1.0, 1.0))

        # ── Node features ─────────────────────────────────────────────────
        bh_cx = cx[bh_idx]
        bh_cy = cy[bh_idx]
        node_feats = np.zeros((_NUM_NODES, NODE_DIM), dtype=np.float32)
        for i, r in enumerate(roles):
            spd = float(p_speed[r])
            node_feats[i] = [
                cx[r] / self.W,
                cy[r] / self.H,
                np.clip(p_vel[r, 0] / self.max_speed, -1.0, 1.0),
                np.clip(p_vel[r, 1] / self.max_speed, -1.0, 1.0),
                np.clip(spd / self.max_speed, 0.0, 1.0),
                w[r] / self.W,
                h[r] / self.H,
                # New: BH-relative position (0,0 for BH node itself)
                (cx[r] - bh_cx) / self.W,
                (cy[r] - bh_cy) / self.H,
                # New: unit velocity heading (direction independent of speed)
                float(np.clip(p_vel[r, 0] / (spd + 1e-6), -1.0, 1.0)),
                float(np.clip(p_vel[r, 1] / (spd + 1e-6), -1.0, 1.0)),
            ]

        # ── Edge features ─────────────────────────────────────────────────
        edge_feats = np.zeros((_NUM_EDGES, EDGE_DIM), dtype=np.float32)
        for e, (a, b) in enumerate(_EDGE_PAIRS):
            ra, rb = roles[a], roles[b]
            dx = (cx[rb] - cx[ra]) / self.W
            dy = (cy[rb] - cy[ra]) / self.H
            dvx = np.clip((p_vel[rb, 0] - p_vel[ra, 0]) / self.max_speed, -1.0, 1.0)
            dvy = np.clip((p_vel[rb, 1] - p_vel[ra, 1]) / self.max_speed, -1.0, 1.0)
            dist = float(np.sqrt(dx**2 + dy**2))

            # Approach rate: positive = A is moving toward B.
            ab_unit = np.array([dx, dy]) / (dist + 1e-6)
            va_norm = p_vel[ra] / self.max_speed
            approach_rate = float(np.clip(np.dot(va_norm, ab_unit), -1.0, 1.0))

            # Speed ratio: >1 means A is faster than B (screener planted if <1 on S→BH edge).
            speed_ratio = float(np.clip(p_speed[ra] / (p_speed[rb] + 1e-6), 0.0, 10.0) / 10.0)

            # New: frame-level scalars appended to every edge.
            edge_feats[e] = [dx, dy, dvx, dvy, dist, approach_rate, speed_ratio,
                             blocker_score, screen_angle_cos]

        return GraphFrame(node_features=node_feats, edge_features=edge_feats, valid=True)
