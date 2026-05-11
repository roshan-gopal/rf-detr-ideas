# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Temporal model over per-frame graph feature vectors (pick-and-roll detection).

Flattens each :class:`~rfdetr.graph.GraphFrame` to length
:data:`~rfdetr.graph.GRAPH_FEATURE_DIM` via :func:`~rfdetr.graph.flatten_graph_frame`,
stacks a clip as ``(B, T, D)``, and runs a :class:`torch.nn.TransformerEncoder`
with sinusoidal positional encoding.

**Frame-level mode** (default): the classifier returns one logit **per frame**
``(B, T)`` — the model predicts whether a pick-and-roll is occurring at each
individual timestep, using full temporal context from attention across all frames.
Use a padding mask to exclude invalid frames from the loss.

**Clip-level mode** (``frame_level=False``): pooled mean over time → one logit
per clip ``(B,)``.  Kept for backwards compatibility and multi-clip baselines.

Example (frame-level)::

    from rfdetr.graph import GraphFrame, GRAPH_FEATURE_DIM
    from rfdetr.temporal import (
        PickAndRollTemporalClassifier,
        encode_graph_sequence,
    )

    classifier = PickAndRollTemporalClassifier(
        embed_dim=GRAPH_FEATURE_DIM,
        num_heads=3,
        frame_level=True,   # default
    )

    # ``graphs`` is a list of ``GraphFrame`` for one clip (length T).
    x, padding_mask = encode_graph_sequence(graphs)
    logits = classifier(x, src_key_padding_mask=padding_mask)  # shape (1, T)
    # loss: BCEWithLogitsLoss on logits[~padding_mask] vs per-frame labels
"""

from __future__ import annotations

__all__ = [
    "encode_graph_sequence",
    "PickAndRollTemporalEncoder",
    "PickAndRollTemporalClassifier",
    "SinusoidalPositionEncoding",
]

import math

import torch
import torch.nn as nn

from rfdetr.graph import GRAPH_FEATURE_DIM, GraphFrame, flatten_graph_frame


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal PE added to ``(B, T, d_model)`` sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        # Odd d_model: even-indexed columns are one more than odd-indexed; split div_term for cos.
        pe[:, 0::2] = torch.sin(position * div_term.unsqueeze(0))
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1].unsqueeze(0))
        else:
            pe[:, 1::2] = torch.cos(position * div_term.unsqueeze(0))
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x``.

        Args:
            x: Tensor of shape ``(B, T, d_model)``.

        Returns:
            Same shape as ``x`` with PE added and dropout applied.
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"sequence length {seq_len} exceeds max positional encoding length {self.pe.size(1)}"
            )
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class PickAndRollTemporalEncoder(nn.Module):
    """Transformer encoder over time on flattened graph features.

    Expects input shape ``(B, T, embed_dim)`` — one token per frame (default
    ``embed_dim=GRAPH_FEATURE_DIM``).

    Args:
        embed_dim: Per-frame channel size (default ``GRAPH_FEATURE_DIM`` =
            flattened graph). Must be divisible by ``num_heads`` (e.g.
            ``num_heads=3`` when ``embed_dim`` matches the flattened dim).
        num_heads: Attention heads.
        num_layers: Stacked :class:`torch.nn.TransformerEncoderLayer` count.
        dim_feedforward: FFN hidden size inside each layer.
        dropout: Dropout on attention and FFN.
        max_seq_len: Maximum sequence length for positional encoding.
        activation: FFN activation (passed to ``TransformerEncoderLayer``).
    """

    def __init__(
        self,
        embed_dim: int = GRAPH_FEATURE_DIM,
        num_heads: int = 3,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.pos_encoding = SinusoidalPositionEncoding(
            d_model=embed_dim,
            max_len=max_seq_len,
            dropout=dropout,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run transformer over sequence, returning full ``(B, T, embed_dim)`` output.

        Args:
            x: Float tensor ``(B, T, embed_dim)``.
            src_key_padding_mask: Optional bool tensor ``(B, T)``.  ``True``
                marks positions to **ignore** (PyTorch convention).

        Returns:
            Tensor of shape ``(B, T, embed_dim)`` — each frame is contextualized
            by all other frames in the clip via self-attention.
        """
        if x.dim() != 3:
            raise ValueError(f"expected x shape (B, T, D), got {tuple(x.shape)}")
        h = self.pos_encoding(x)
        return self.transformer(h, src_key_padding_mask=src_key_padding_mask)


class PickAndRollTemporalClassifier(nn.Module):
    """Transformer encoder followed by a per-frame (or clip-level) linear head.

    Args:
        encoder: Optional pre-built :class:`PickAndRollTemporalEncoder`.  If
            omitted, one is constructed from ``**encoder_kwargs``.
        frame_level: If ``True`` (default), output shape is ``(B, T)`` — one
            logit per frame.  If ``False``, mean-pool over valid timesteps and
            return ``(B,)`` — one logit per clip.
        **encoder_kwargs: Forwarded to :class:`PickAndRollTemporalEncoder` when
            ``encoder`` is ``None``.
    """

    def __init__(
        self,
        encoder: PickAndRollTemporalEncoder | None = None,
        frame_level: bool = True,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else PickAndRollTemporalEncoder(**encoder_kwargs)
        self.head = nn.Linear(self.encoder.embed_dim, 1)
        self.frame_level = frame_level

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits for pick-and-roll prediction.

        Args:
            x: ``(B, T, embed_dim)``.
            src_key_padding_mask: Optional ``(B, T)`` bool mask; ``True``
                marks frames to ignore.

        Returns:
            ``(B, T)`` per-frame logits when ``frame_level=True``, or ``(B,)``
            clip logits when ``frame_level=False``.
        """
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, D)
        logits = self.head(h).squeeze(-1)                                # (B, T)
        if self.frame_level:
            return logits
        # Clip-level: masked mean pool then scalar
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).to(dtype=h.dtype)            # (B, T)
            summed = (logits * valid).sum(dim=1)
            denom = valid.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        return logits.mean(dim=1)


def encode_graph_sequence(
    graphs: list[GraphFrame],
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ``GraphFrame`` list into one batch item ``(1, T, GRAPH_FEATURE_DIM)``.

    Invalid graphs produce a zero vector (see :func:`~rfdetr.graph.flatten_graph_frame`)
    and are marked in ``src_key_padding_mask`` so pooling and attention can skip them.

    Args:
        graphs: Ordered frames for one clip.
        device: Tensor device (default: CPU).

    Returns:
        ``(x, src_key_padding_mask)`` where ``x`` is ``(1, T, GRAPH_FEATURE_DIM)``
        and ``src_key_padding_mask`` is ``(1, T)`` bool, ``True`` where the
        frame should be ignored (invalid graph).
    """
    if len(graphs) == 0:
        raise ValueError("graphs must be non-empty")
    dev = device or torch.device("cpu")
    rows: list[torch.Tensor] = []
    ignore: list[bool] = []
    for g in graphs:
        vec = flatten_graph_frame(g)
        rows.append(torch.from_numpy(vec).to(dev))
        ignore.append(not g.valid)
    x = torch.stack(rows, dim=0).unsqueeze(0)
    src_key_padding_mask = torch.tensor([ignore], dtype=torch.bool, device=dev)
    return x, src_key_padding_mask
