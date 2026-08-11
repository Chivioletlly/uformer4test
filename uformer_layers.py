"""Small, dependency-free equivalents of the timm layers used by Uformer.

The official Uformer source imports these three helpers from timm.  Keeping the
implementations local avoids pulling the rest of timm into the frozen AIO3
environment while preserving the same stochastic-depth and initialization
semantics.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def to_2tuple(value):
    """Return an existing iterable or repeat a scalar twice, as timm does."""

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return value
    return (value, value)


def drop_path(
    tensor: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Drop residual paths independently for each sample."""

    if drop_prob == 0.0 or not training:
        return tensor
    keep_prob = 1.0 - float(drop_prob)
    if keep_prob <= 0.0:
        raise ValueError("drop_prob must be smaller than 1.0")
    shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    random_tensor.floor_()
    return tensor.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Per-sample stochastic depth used by the official Uformer blocks."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return drop_path(tensor, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"


def trunc_normal_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Initialize with the PyTorch truncated normal used by current timm."""

    return nn.init.trunc_normal_(tensor, std=std)
