"""AIO3-v1 adapter for the official Uformer-B restoration network."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from model import Uformer


UPSTREAM_REPOSITORY = "https://github.com/ZhendongWang6/Uformer"
UPSTREAM_COMMIT = "65fc970a8ffc09605faca74ed016ee93c9ad8a36"
MODEL_NAME = "uformer_b"
MODEL_VARIANT = "Uformer_B"
INPUT_MULTIPLE = 128
DEPTHS: Tuple[int, ...] = (1, 2, 8, 8, 2, 8, 8, 2, 1)
NUM_HEADS: Tuple[int, ...] = (1, 2, 4, 8, 16, 16, 8, 4, 2)


def _as_int_tuple(values: Sequence[object], *, name: str) -> Tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def expected_checkpoint_metadata(model_config: Mapping[str, object]) -> Dict[str, object]:
    """Build the architecture identity stored in every AIO3 checkpoint."""

    return {
        "architecture_version": 1,
        "model_name": MODEL_NAME,
        "model_variant": MODEL_VARIANT,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "img_size": int(model_config["img_size"]),
        "in_chans": int(model_config["in_chans"]),
        "dd_in": int(model_config["dd_in"]),
        "embed_dim": int(model_config["embed_dim"]),
        "depths": list(_as_int_tuple(model_config["depths"], name="depths")),
        "num_heads": list(
            _as_int_tuple(model_config["num_heads"], name="num_heads")
        ),
        "win_size": int(model_config["win_size"]),
        "mlp_ratio": float(model_config["mlp_ratio"]),
        "qkv_bias": bool(model_config["qkv_bias"]),
        "drop_rate": float(model_config["drop_rate"]),
        "attn_drop_rate": float(model_config["attn_drop_rate"]),
        "drop_path_rate": float(model_config["drop_path_rate"]),
        "patch_norm": bool(model_config["patch_norm"]),
        "use_checkpoint": bool(model_config["use_checkpoint"]),
        "token_projection": str(model_config["token_projection"]),
        "token_mlp": str(model_config["token_mlp"]),
        "shift_flag": bool(model_config["shift_flag"]),
        "modulator": bool(model_config["modulator"]),
        "cross_modulator": bool(model_config["cross_modulator"]),
        "input_multiple": int(model_config["input_multiple"]),
        "padding_mode": "zero_right_bottom_to_square_multiple_128",
        "output_mode": "input_plus_unbounded_signed_residual",
        "pretrained": False,
    }


def validate_uformer_config(model_config: Mapping[str, object]) -> None:
    """Reject architecture drift from the frozen Uformer-B comparison model."""

    required = {
        "name": MODEL_NAME,
        "variant": MODEL_VARIANT,
        "img_size": 128,
        "in_chans": 3,
        "dd_in": 3,
        "embed_dim": 32,
        "depths": list(DEPTHS),
        "num_heads": list(NUM_HEADS),
        "win_size": 8,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "patch_norm": True,
        "use_checkpoint": False,
        "token_projection": "linear",
        "token_mlp": "leff",
        "shift_flag": True,
        "modulator": True,
        "cross_modulator": False,
        "input_multiple": INPUT_MULTIPLE,
        "pretrained": False,
    }
    mismatches = []
    for key, expected in required.items():
        actual = model_config.get(key)
        if key in {"depths", "num_heads"} and actual is not None:
            actual = [int(value) for value in actual]
        if actual != expected:
            mismatches.append(f"{key}: {actual!r} != {expected!r}")
    if mismatches:
        raise ValueError("Frozen Uformer-B config mismatch: " + "; ".join(mismatches))


def validate_uformer_checkpoint(
    metadata: Mapping[str, object],
    model_config: Mapping[str, object],
) -> None:
    """Reject checkpoints that do not exactly identify the frozen architecture."""

    expected = expected_checkpoint_metadata(model_config)
    actual = dict(metadata)
    if actual != expected:
        keys = sorted(set(actual) | set(expected))
        details = [
            f"{key}: checkpoint={actual.get(key)!r}, expected={expected.get(key)!r}"
            for key in keys
            if actual.get(key) != expected.get(key)
        ]
        raise RuntimeError("Incompatible Uformer checkpoint metadata: " + "; ".join(details))


class AIO3Uformer(nn.Module):
    """Uformer-B with deterministic native-resolution padding and cropping.

    The official implementation reconstructs spatial dimensions with ``sqrt``
    and therefore requires square inputs.  Its four downsampling stages and
    8x8 attention windows additionally require a side divisible by 128.  The
    adapter pads only the bottom/right with zeros, runs the unchanged network,
    and crops the unbounded prediction back to the original HxW.
    """

    def __init__(self, model_config: Mapping[str, object]) -> None:
        super().__init__()
        validate_uformer_config(model_config)
        self.model_config = dict(model_config)
        self.input_multiple = int(model_config["input_multiple"])
        self.backbone = Uformer(
            img_size=int(model_config["img_size"]),
            in_chans=int(model_config["in_chans"]),
            dd_in=int(model_config["dd_in"]),
            embed_dim=int(model_config["embed_dim"]),
            depths=list(_as_int_tuple(model_config["depths"], name="depths")),
            num_heads=list(
                _as_int_tuple(model_config["num_heads"], name="num_heads")
            ),
            win_size=int(model_config["win_size"]),
            mlp_ratio=float(model_config["mlp_ratio"]),
            qkv_bias=bool(model_config["qkv_bias"]),
            drop_rate=float(model_config["drop_rate"]),
            attn_drop_rate=float(model_config["attn_drop_rate"]),
            drop_path_rate=float(model_config["drop_path_rate"]),
            patch_norm=bool(model_config["patch_norm"]),
            use_checkpoint=bool(model_config["use_checkpoint"]),
            token_projection=str(model_config["token_projection"]),
            token_mlp=str(model_config["token_mlp"]),
            shift_flag=bool(model_config["shift_flag"]),
            modulator=bool(model_config["modulator"]),
            cross_modulator=bool(model_config["cross_modulator"]),
        )

    def checkpoint_metadata(self) -> Dict[str, object]:
        return expected_checkpoint_metadata(self.model_config)

    def padded_side(self, height: int, width: int) -> int:
        if height <= 0 or width <= 0:
            raise ValueError(f"Spatial dimensions must be positive, got {(height, width)}")
        return int(math.ceil(max(height, width) / self.input_multiple) * self.input_multiple)

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        if degraded.ndim != 4:
            raise ValueError(f"Expected BCHW input, got shape {tuple(degraded.shape)}")
        if degraded.shape[1] != 3:
            raise ValueError(f"Expected 3 RGB channels, got {degraded.shape[1]}")
        height, width = int(degraded.shape[-2]), int(degraded.shape[-1])
        side = self.padded_side(height, width)
        pad_right = side - width
        pad_bottom = side - height
        if pad_right or pad_bottom:
            padded = F.pad(
                degraded,
                (0, pad_right, 0, pad_bottom),
                mode="constant",
                value=0.0,
            )
        else:
            padded = degraded
        restored_padded = self.backbone(padded)
        return restored_padded[..., :height, :width]
