import sys
from pathlib import Path

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from aio3_runner.models import build_model, model_parameter_counts, validate_model_checkpoint
from aio3_runner.runtime import build_run_config
from uformer_aio3_model import AIO3Uformer, validate_uformer_config


def _frozen_config():
    return build_run_config(
        run_kind="smoke",
        seed=3407,
        run_name="uformer-unit",
        run_dir=Path("/tmp/aio3-v1/uformer/uformer-unit"),
        manifest_hashes={"train.jsonl": "unit"},
        repository_state={
            "commit": "unit-commit",
            "dirty": False,
            "protocol_document_sha256": "unit-protocol",
        },
        num_workers=0,
        wandb_run_id="unit-run-id",
        wandb_mode="disabled",
        wandb_entity=None,
    )


class _RecordAndOffset(nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = float(offset)
        self.seen = None

    def forward(self, tensor):
        self.seen = tensor.detach().clone()
        return tensor + self.offset


def _adapter_without_full_backbone(offset: float = -2.0):
    adapter = AIO3Uformer.__new__(AIO3Uformer)
    nn.Module.__init__(adapter)
    adapter.input_multiple = 128
    adapter.backbone = _RecordAndOffset(offset)
    return adapter


def _assert_raises(exception_type, message, function, *args):
    try:
        function(*args)
    except exception_type as error:
        assert message in str(error)
    else:
        raise AssertionError(f"Expected {exception_type.__name__}: {message}")


def test_frozen_uformer_b_parameter_count_and_checkpoint_identity():
    config = _frozen_config()
    model = build_model(config)
    assert model_parameter_counts(model) == (50880946, 50880946)
    validate_model_checkpoint(model.checkpoint_metadata(), config)
    incompatible = dict(model.checkpoint_metadata())
    incompatible["embed_dim"] = 16
    _assert_raises(
        RuntimeError,
        "Incompatible Uformer checkpoint",
        validate_model_checkpoint,
        incompatible,
        config,
    )


def test_adapter_pads_to_square_multiple_and_crops_without_clamping():
    model = _adapter_without_full_backbone(offset=-2.0)
    degraded = torch.ones(1, 3, 127, 191)
    restored = model(degraded)
    assert restored.shape == degraded.shape
    torch.testing.assert_close(restored, torch.full_like(restored, -1.0))
    assert model.backbone.seen.shape == (1, 3, 256, 256)
    torch.testing.assert_close(model.backbone.seen[..., :127, :191], degraded)
    assert torch.count_nonzero(model.backbone.seen[..., 127:, :]) == 0
    assert torch.count_nonzero(model.backbone.seen[..., :, 191:]) == 0


def test_frozen_config_rejects_architecture_drift():
    model_config = dict(_frozen_config()["model"])
    model_config["pretrained"] = True
    _assert_raises(ValueError, "pretrained", validate_uformer_config, model_config)


def test_full_uformer_b_forward_backward_is_finite_and_unclamped():
    torch.manual_seed(3407)
    model = build_model(_frozen_config()).train()
    degraded = torch.rand(1, 3, 128, 128)
    target = torch.rand_like(degraded)
    restored = model(degraded)
    assert restored.shape == degraded.shape
    assert torch.isfinite(restored).all()
    assert bool((restored < 0.0).any() or (restored > 1.0).any())
    loss = (restored - target).abs().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


if __name__ == "__main__":
    tests = (
        test_frozen_uformer_b_parameter_count_and_checkpoint_identity,
        test_adapter_pads_to_square_multiple_and_crops_without_clamping,
        test_frozen_config_rejects_architecture_drift,
        test_full_uformer_b_forward_backward_is_finite_and_unclamped,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")
