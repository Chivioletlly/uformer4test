"""CUDA/BF16 acceptance test for the frozen AIO3 Uformer-B adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from aio3_runner.models import build_model, model_parameter_counts
from aio3_runner.runtime import build_run_config


def _config():
    return build_run_config(
        run_kind="smoke",
        seed=3407,
        run_name="cuda-acceptance",
        run_dir=Path("/tmp/aio3-v1/uformer/cuda-acceptance"),
        manifest_hashes={"train.jsonl": "acceptance"},
        repository_state={
            "commit": "acceptance",
            "dirty": False,
            "protocol_document_sha256": "acceptance",
        },
        num_workers=0,
        wandb_run_id="acceptance",
        wandb_mode="disabled",
        wandb_entity=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-native-resolution",
        action="store_true",
        help="Skip the more memory-intensive 127x191 -> 256x256 padding check.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required")

    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    device = torch.device("cuda", 0)
    model = build_model(_config()).to(device).train()
    counts = model_parameter_counts(model)
    if counts != (50880946, 50880946):
        raise RuntimeError(f"Unexpected parameter counts: {counts}")

    degraded = torch.rand(1, 3, 128, 128, device=device)
    target = torch.rand_like(degraded)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        restored = model(degraded)
        loss = (restored - target).abs().mean()
    if restored.shape != degraded.shape or not torch.isfinite(loss):
        raise RuntimeError("BF16 forward produced an invalid result")
    loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if missing or nonfinite:
        raise RuntimeError(
            f"Invalid gradients: missing={missing[:10]}, nonfinite={nonfinite[:10]}"
        )

    model.eval()
    if not args.skip_native_resolution:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            native = model(torch.rand(1, 3, 127, 191, device=device))
        if native.shape != (1, 3, 127, 191) or not torch.isfinite(native).all():
            raise RuntimeError("Native-resolution padding/crop check failed")

    allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"parameters: {counts[0]}")
    print(f"max_memory_allocated_gib: {allocated:.3f}")
    print("Uformer-B FP32 parameters / BF16 autocast forward-backward: PASS")


if __name__ == "__main__":
    main()
