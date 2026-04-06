# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Player tracking with velocity estimation for RF-DETR NBA analytics.

Wraps ``supervision.ByteTrack`` to maintain consistent player IDs across
frames and compute per-player velocity vectors (pixels/second) from a
rolling position history.  Velocity is attached to ``sv.Detections.data``
under the key ``"velocity"`` as a float32 array of shape ``(N, 2)``
representing ``[vx, vy]`` for each tracked detection.

Example::

    import supervision as sv
    from rfdetr import RFDETRLarge
    from rfdetr.tracking import PlayerTracker

    model = RFDETRLarge()
    tracker = PlayerTracker(fps=30.0)

    for frame in video_frames:
        detections = model.predict(frame, threshold=0.5)
        tracked = tracker.update(detections)

        # tracked.tracker_id — stable player IDs
        # tracked.data["velocity"] — shape (N, 2), [vx, vy] pixels/sec
        # tracked.data["speed"]    — shape (N,),  scalar pixels/sec
"""

from __future__ import annotations

__all__ = ["PlayerTracker"]

from collections import defaultdict, deque
from typing import Deque

import numpy as np
import supervision as sv


class PlayerTracker:
    """Frame-by-frame player tracker with velocity estimation.

    On each call to :meth:`update`, raw RF-DETR detections are passed to
    ``ByteTrack`` which assigns stable track IDs.  The centre of each
    tracked bounding box is appended to that track's position history, and
    velocity is estimated from the displacement between the two most recent
    positions multiplied by ``fps``.

    Detections that have been tracked for only one frame receive a zero
    velocity vector.  Velocity is expressed in **pixels per second** relative
    to the original (unresized) image coordinate space.

    Args:
        fps: Frame rate of the source video.  Used to convert per-frame
            displacement into pixels per second.
        history_len: Number of past positions to retain per track.  The
            oldest positions are dropped automatically once the deque is
            full.  Only the two most recent positions are used for velocity
            estimation; a longer history is kept so that downstream modules
            can access richer motion context if needed.
        byte_track_kwargs: Optional keyword arguments forwarded verbatim to
            ``supervision.ByteTrack.__init__``.  Common knobs include
            ``track_activation_threshold``, ``lost_track_buffer``, and
            ``minimum_consecutive_frames``.
    """

    def __init__(
        self,
        fps: float = 30.0,
        history_len: int = 5,
        **byte_track_kwargs,
    ) -> None:
        self.fps = fps
        self.history_len = history_len
        self._tracker = sv.ByteTrack(**byte_track_kwargs)
        self._position_history: dict[int, Deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=history_len)
        )

    def reset(self) -> None:
        """Reset tracker state and all position histories.

        Call this when switching to a new video clip so stale track IDs
        and positions from the previous clip do not bleed into the new one.
        """
        self._tracker.reset()
        self._position_history.clear()

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker with new frame detections and attach velocity.

        Args:
            detections: Raw ``sv.Detections`` from ``RFDETR.predict`` for
                the current frame.  ``tracker_id`` need not be set; it will
                be populated by ``ByteTrack``.

        Returns:
            A new ``sv.Detections`` object with ``tracker_id`` set and two
            additional entries in ``.data``:

            * ``"velocity"``: ``np.ndarray`` of shape ``(N, 2)``, dtype
              ``float32``.  Each row is ``[vx, vy]`` in pixels per second.
            * ``"speed"``: ``np.ndarray`` of shape ``(N,)``, dtype
              ``float32``.  Euclidean magnitude of the velocity vector.
        """
        tracked = self._tracker.update_with_detections(detections)

        if len(tracked) == 0:
            tracked.data["velocity"] = np.empty((0, 2), dtype=np.float32)
            tracked.data["speed"] = np.empty((0,), dtype=np.float32)
            return tracked

        # Compute box centres in original image coordinates.
        centres = np.stack(
            [
                (tracked.xyxy[:, 0] + tracked.xyxy[:, 2]) / 2,
                (tracked.xyxy[:, 1] + tracked.xyxy[:, 3]) / 2,
            ],
            axis=1,
        )  # (N, 2)

        velocities = np.zeros((len(tracked), 2), dtype=np.float32)
        for i, track_id in enumerate(tracked.tracker_id):
            cx, cy = float(centres[i, 0]), float(centres[i, 1])
            history = self._position_history[track_id]
            if len(history) >= 1:
                prev_cx, prev_cy = history[-1]
                velocities[i, 0] = (cx - prev_cx) * self.fps
                velocities[i, 1] = (cy - prev_cy) * self.fps
            history.append((cx, cy))

        tracked.data["velocity"] = velocities
        tracked.data["speed"] = np.linalg.norm(velocities, axis=1).astype(np.float32)
        return tracked
