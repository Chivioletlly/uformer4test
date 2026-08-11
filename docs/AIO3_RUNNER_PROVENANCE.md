# AIO3-v1 runner provenance for Uformer-B

The common AIO3 runner was copied from
`Chivioletlly/DCNv4fortest@6a2c14f92770506fe3b2558ec4072037189b1ea9`, the
reference training commit named by `AIO3_MODEL_COMPARISON_STANDARD.md`.

`aio3_runner/common_runner_sha256.json` freezes the byte-level SHA256 values of
the model-neutral checkpoint, data, manifest, noise, schedule, metric,
validation, and formal-evaluation modules.  `tests/test_runner_provenance.py`
checks those hashes and fails if a future change silently alters common
comparison behavior.

The model-specific integration is limited to:

- `uformer_aio3_model.py` and `aio3_runner/models.py`: architecture construction,
  native-resolution padding/cropping, parameter counting, and checkpoint
  metadata validation;
- `aio3_runner/runtime.py`: the frozen Uformer-B config, output/model names, and
  model environment metadata;
- `aio3_runner/train.py`, `aio3_runner/training.py`, and
  `aio3_runner/evaluate.py`: model builder/validator imports and Uformer-specific
  command descriptions;
- `aio3_runner/monitoring.py`: W&B tags and artifact names.

The balanced sampler, online Gaussian noise, synchronized augmentation, pixel
L1 loss, AdamW update, warmup-cosine schedule, metric implementation, checkpoint
selection, validation, and formal test logic remain the reference behavior.

The upstream Uformer architecture is frozen at
`ZhendongWang6/Uformer@65fc970a8ffc09605faca74ed016ee93c9ad8a36`.  Formal
AIO3 runs use the official `Uformer_B` configuration and random initialization;
pretrained weights are never loaded.
