import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_model_neutral_runner_files_match_frozen_reference():
    manifest_path = REPOSITORY_ROOT / "aio3_runner" / "common_runner_sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["reference_commit"] == "6a2c14f92770506fe3b2558ec4072037189b1ea9"
    for relative, expected in manifest["sha256"].items():
        path = REPOSITORY_ROOT / relative
        assert path.is_file(), relative
        assert _sha256(path) == expected, relative


if __name__ == "__main__":
    test_model_neutral_runner_files_match_frozen_reference()
    print("test_model_neutral_runner_files_match_frozen_reference: PASS")
