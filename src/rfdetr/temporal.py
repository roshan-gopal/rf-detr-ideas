# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Temporal attention over per-frame graph embeddings (pick-and-roll clip model).

Stacks outputs of :class:`~rfdetr.graph.PickAndRollGNN` into a sequence ``(B, T, D)``
and runs a :class:`torch.nn.TransformerEncoder` with sinusoidal positional
encoding.  Pooling produces a clip-level vector for classification or downstream
heads.

Example::

    from rfdetr.graph import GraphFrame, PickAndRollGNN
    from rfdetr.temporal import (
        PickAndRollTemporalClassifier,
        PickAndRollTemporalEncoder,
        encode_graph_sequence,
    )

    gnn = PickAndRollGNN(out_dim=32)
    temporal = PickAndRollTemporalEncoder(embed_dim=32)
    classifier = PickAndRollTemporalClassifier(embed_dim=32)  # encoder + linear

    # ``graphs`` is a list of ``GraphFrame`` for one clip (length T).
    x, padding_mask = encode_graph_sequence(gnn, graphs)
    logit = classifier(x, src_key_padding_mask=padding_mask)  # shape (1,)
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

from rfdetr.graph import GraphFrame, PickAndRollGNN


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
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
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
    """Transformer encoder over time on graph embeddings.

    Expects input shape ``(B, T, embed_dim)`` — one token per frame.  When
    ``src_key_padding_mask`` is provided (``True`` = ignore position), pooling
    uses a masked mean over valid timesteps only.

    Args:
        embed_dim: Channel size (must match :class:`~rfdetr.graph.PickAndRollGNN`
            ``out_dim``).  Must be divisible by ``num_heads``.
        num_heads: Attention heads.
        num_layers: Stacked :class:`torch.nn.TransformerEncoderLayer` count.
        dim_feedforward: FFN hidden size inside each layer.
        dropout: Dropout on attention and FFN.
        max_seq_len: Maximum sequence length for positional encoding.
        activation: FFN activation (passed to ``TransformerEncoderLayer``).
    """

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 4,
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
        """Encode a batch of embedding sequences.

        Args:
            x: Float tensor ``(B, T, embed_dim)``.
            src_key_padding_mask: Optional bool tensor ``(B, T)``.  ``True``
                marks positions to **ignore** (PyTorch convention, same as
                :class:`torch.nn.TransformerEncoder`).

        Returns:
            Clip embedding of shape ``(B, embed_dim)``.
        """
        if x.dim() != 3:
            raise ValueError(f"expected x shape (B, T, D), got {tuple(x.shape)}")
        h = self.pos_encoding(x)
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).to(dtype=h.dtype).unsqueeze(-1)
            summed = (h * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        return h.mean(dim=1)


class PickAndRollTemporalClassifier(nn.Module):
    """Temporal encoder followed by a single linear layer (binary logit)."""

    def __init__(
        self,
        encoder: PickAndRollTemporalEncoder | None = None,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else PickAndRollTemporalEncoder(**encoder_kwargs)
        self.head = nn.Linear(self.encoder.embed_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-batch logits for "pick-and-roll clip".

        Args:
            x: ``(B, T, embed_dim)``.
            src_key_padding_mask: Optional ``(B, T)`` padding mask.

        Returns:
            Logits of shape ``(B,)`` (squeeze last dim of linear output).
        """
        z = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.head(z).squeeze(-1)


def encode_graph_sequence(
    gnn: PickAndRollGNN,
    graphs: list[GraphFrame],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ``GraphFrame`` list into one batch item ``(1, T, D)`` with padding mask.

    Invalid graphs still produce a zero vector from ``gnn`` but are marked as
    padding so the temporal encoder can exclude them from masked pooling and
    (if desired) from attention via ``src_key_padding_mask``.

    Args:
        gnn: Per-frame graph MLP.
        graphs: Ordered frames for one clip.

    Returns:
        ``(x, src_key_padding_mask)`` where ``x`` is ``(1, T, out_dim)`` and
        ``src_key_padding_mask`` is ``(1, T)`` bool, ``True`` where the frame
        should be ignored (invalid graph).
    """
    if len(graphs) == 0:
        raise ValueError("graphs must be non-empty")
    device = next(gnn.parameters()).device
    rows: list[torch.Tensor] = []
    ignore: list[bool] = []
    for g in graphs:
        e = gnn(g)
        if e.device != device:
            e = e.to(device)
        rows.append(e)
        ignore.append(not g.valid)
    x = torch.stack(rows, dim=0).unsqueeze(0)
    src_key_padding_mask = torch.tensor([ignore], dtype=torch.bool, device=device)
    return x, src_key_padding_mask
