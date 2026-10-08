"""Download the configured GGUF model from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "unsloth/Qwen3.5-2B-GGUF"
DEFAULT_MODEL_FILE = "Qwen3.5-2B-Q5_K_M.gguf"
DEFAULT_REVISION = "main"
DEFAULT_MODEL_DIR = "models"
CHUNK_SIZE = 8 * 1024 * 1024

# The tested default model is pinned so an incomplete or changed file is rejected.
PINNED_MODELS = {
    (DEFAULT_MODEL_ID, DEFAULT_MODEL_FILE, DEFAULT_REVISION): (
        1_435_238_656,
        "1885b3a9195f8cc09da9a7a7a75afdc1e8d5cbf9fc4a499c3961dddea37098ac",
    )
}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    filename: str
    revision: str
    model_dir: Path
    token: str | None

    @property
    def destination(self) -> Path:
        return self.model_dir.joinpath(*PurePosixPath(self.filename).parts)

    @classmethod
    def from_environment(cls, project_root: Path = PROJECT_ROOT) -> "ModelConfig":
        filename = os.getenv("HF_MODEL_FILE", DEFAULT_MODEL_FILE).strip()
        filename_path = PurePosixPath(filename)
        if filename_path.is_absolute() or ".." in filename_path.parts:
            raise ValueError("HF_MODEL_FILE must be a relative Hugging Face filename.")

        configured_dir = Path(os.getenv("HF_MODEL_DIR", DEFAULT_MODEL_DIR))
        model_dir = (
            configured_dir
            if configured_dir.is_absolute()
            else project_root / configured_dir
        ).resolve()
        project_root = project_root.resolve()
        if not model_dir.is_relative_to(project_root):
            raise ValueError("HF_MODEL_DIR must stay inside the project directory.")

        return cls(
            model_id=os.getenv("HF_MODEL_ID", DEFAULT_MODEL_ID).strip(),
            filename=filename,
            revision=os.getenv("HF_MODEL_REVISION", DEFAULT_REVISION).strip(),
            model_dir=model_dir,
            token=os.getenv("HF_TOKEN") or None,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pinned_model(config: ModelConfig, path: Path) -> None:
    expected = PINNED_MODELS.get(
        (config.model_id, config.filename, config.revision)
    )
    if expected is None:
        return

    expected_size, expected_hash = expected
    if path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Unexpected model size: {path.stat().st_size}; expected {expected_size}."
        )
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Unexpected SHA-256: {actual_hash}; expected {expected_hash}."
        )


def download(config: ModelConfig) -> Path:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=config.model_id,
            filename=config.filename,
            revision=config.revision,
            local_dir=config.model_dir,
            token=config.token,
        )
    )
    verify_pinned_model(config, downloaded)
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--show-config",
        action="store_true",
        help="Show the resolved model settings without downloading anything.",
    )
    output_mode.add_argument(
        "--print-path",
        action="store_true",
        help="Print only the configured destination path without downloading.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    try:
        config = ModelConfig.from_environment()
        if args.print_path:
            print(config.destination)
            return 0
        print(f"Model repository: {config.model_id}")
        print(f"Model file:       {config.filename}")
        print(f"Revision:         {config.revision}")
        print(f"Destination:      {config.destination}")
        print(f"HF token set:     {'yes' if config.token else 'no (public access)'}")
        if args.show_config:
            return 0

        downloaded = download(config)
        print(f"Model ready: {downloaded} ({downloaded.stat().st_size} bytes)")
    except (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
