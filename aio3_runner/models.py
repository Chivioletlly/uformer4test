"""Frozen Uformer-B adapter for AIO3-v1."""

from __future__ import annotations

from typing import Mapping, Tuple

import torch

from uformer_aio3_model import (
    AIO3Uformer,
    validate_uformer_checkpoint,
)


def build_model(config: Mapping[str, object]) -> AIO3Uformer:
    model_config = dict(config["model"])
    model = AIO3Uformer(model_config)
    validate_uformer_checkpoint(model.checkpoint_metadata(), model_config)
    return model


def validate_model_checkpoint(
    metadata: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    validate_uformer_checkpoint(metadata, dict(config["model"]))


def model_parameter_counts(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
