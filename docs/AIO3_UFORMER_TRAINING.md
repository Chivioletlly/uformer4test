# AIO3-v1 Uformer-B training runbook

This runbook trains the official Uformer-B architecture for a direct AIO3-v1
comparison with the completed DCNv4 U-Net baseline.  The formal model is fixed
to 50,880,946 trainable parameters and starts from its native random
initialization.  Do not load any of the pretrained checkpoints linked by the
upstream Uformer README.

## 1. Fixed server paths

```bash
export PROJECT_ROOT=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation
export MODEL_ROOT="${PROJECT_ROOT}/all-in-one-model"
export OUTPUT_ROOT="${PROJECT_ROOT}/outputs/AIO3/aio3-v1"
export MANIFEST_DIR="${OUTPUT_ROOT}/manifests"

cd "${MODEL_ROOT}/uformer"
```

The data root, manifests, checkpoints, predictions, and W&B local files must
remain outside this Git repository.  New Uformer runs are created under
`${OUTPUT_ROOT}/uformer/<run_name>`.

## 2. Update and install the frozen environment

```bash
git pull --ff-only origin main
git status --short --branch
git rev-parse HEAD

conda env create -f environment-aio3.yml
conda activate uformer-aio3

python -c "import torch, torchvision, einops, wandb; print(torch.__version__, torchvision.__version__, einops.__version__, wandb.__version__)"
```

If `uformer-aio3` already exists, activate it instead of recreating it.  The
required W&B version is exactly `0.25.1`.

## 3. Verify the frozen manifests

Reuse the manifests produced for the DCNv4 baseline; do not regenerate or
overwrite them.

```bash
sha256sum \
  "${MANIFEST_DIR}/train.jsonl" \
  "${MANIFEST_DIR}/val.jsonl" \
  "${MANIFEST_DIR}/test.jsonl" \
  "${MANIFEST_DIR}/data_audit.json" \
  "${MANIFEST_DIR}/visual_samples.json"
```

Expected values:

```text
bd153a3b211957184de7b6171d6bc06a48f321b1c571906604d869b1aa19ca7e  train.jsonl
9c66c4c74a0279858ecab33df998b8eb55d6df021d2e59bbd1c253830ab3f50b  val.jsonl
7d80fd0af7aeaac2b6e901e20e71a744d7d705f98641f54913aa278e12c2b63a  test.jsonl
2959e402ecdb76172b9fe9bba3fae13c090348379dcd33898992abdc198e06b8  data_audit.json
62e9f6e761e3db2c23895958f3707414a59baac30840919044fe6bf848ff628b  visual_samples.json
```

Any mismatch stops the comparison.

## 4. Run preflight tests

```bash
pytest -q
python scripts/test_uformer_aio3_cuda.py
```

The CUDA test checks the frozen parameter count, BF16 forward/backward,
finite gradients, and the adapter's 127x191 native-resolution padding/crop.
Record the reported peak allocated memory.  If it OOMs, do not reduce the
effective batch or alter the model; stop and add protocol-compliant gradient
accumulation before training.

## 5. Verify W&B connectivity

```bash
wandb login

python -m aio3_runner.wandb_check \
  --output-root "${OUTPUT_ROOT}" \
  --entity c14150591-sjtu \
  --mode online
```

Confirm that the connectivity run is visible in the `aio3-restoration`
project before starting smoke training.

## 6. Run the 100-step smoke test

```bash
RUN_NAME="uformer-b-smoke-seed3407-$(date -u +%Y%m%d-%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/uformer/${RUN_NAME}"

python -m aio3_runner.train \
  --manifest-dir "${MANIFEST_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-kind smoke \
  --run-name "${RUN_NAME}" \
  --seed 3407 \
  --num-workers 8 \
  --wandb-mode online \
  --wandb-entity c14150591-sjtu
```

After completion, verify:

```bash
cat "${RUN_DIR}/run_state.json"
tail -n 1 "${RUN_DIR}/train_metrics.jsonl" | python -m json.tool
cat "${RUN_DIR}/wandb_state.json"
ls -lh "${RUN_DIR}/checkpoints"
```

The smoke run must finish at step 100, perform the complete 420-image
validation, create `latest.pth` and `best_macro_psnr.pth`, and show finite loss,
gradient norm, throughput, and memory.

## 7. Run the independent pause/resume acceptance test

Create a second smoke run and pause it at step 50:

```bash
RUN_NAME="uformer-b-resume-smoke-seed3407-$(date -u +%Y%m%d-%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/uformer/${RUN_NAME}"

python -m aio3_runner.train \
  --manifest-dir "${MANIFEST_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-kind smoke \
  --run-name "${RUN_NAME}" \
  --seed 3407 \
  --num-workers 8 \
  --wandb-mode online \
  --wandb-entity c14150591-sjtu \
  --pause-at-step 50

python -m aio3_runner.train \
  --resume "${RUN_DIR}/checkpoints/latest.pth"
```

The resumed process must reuse the same W&B run ID and finish at step 100.

## 8. Run the 5,000-step pilot from random initialization

Do not resume from either smoke checkpoint.

```bash
RUN_NAME="uformer-b-pilot-seed3407-$(date -u +%Y%m%d-%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/uformer/${RUN_NAME}"

python -m aio3_runner.train \
  --manifest-dir "${MANIFEST_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-kind pilot \
  --run-name "${RUN_NAME}" \
  --seed 3407 \
  --num-workers 8 \
  --wandb-mode online \
  --wandb-entity c14150591-sjtu
```

Review all 14 fixed validation samples after the pilot.  Check sigma-50 residual
noise, content removed with rain, haze color/brightness shifts, padding-edge
artifacts, NaN/Inf, and abnormal out-of-range prediction fractions.

## 9. Run the 200,000-step formal training from random initialization

Do not resume the pilot.  Start a new run only after pilot acceptance.

```bash
RUN_NAME="uformer-b-formal-seed3407-$(date -u +%Y%m%d-%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/uformer/${RUN_NAME}"

python -m aio3_runner.train \
  --manifest-dir "${MANIFEST_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-kind formal \
  --run-name "${RUN_NAME}" \
  --seed 3407 \
  --num-workers 8 \
  --wandb-mode online \
  --wandb-entity c14150591-sjtu
```

For an interruption, resume only the same run's `latest.pth`:

```bash
python -m aio3_runner.train \
  --resume "${RUN_DIR}/checkpoints/latest.pth"
```

## 10. Run the frozen formal test once

Only after `run_state.json` reports `completed` at step 200000:

```bash
python -m aio3_runner.evaluate \
  --checkpoint "${RUN_DIR}/checkpoints/best_macro_psnr.pth" \
  --num-workers 4
```

Do not run this command on smoke/pilot checkpoints and do not use the test
results for adapter changes or hyperparameter selection.
