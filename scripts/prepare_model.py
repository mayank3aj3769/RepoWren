"""Download, split, and validate the model configured for AirLLM."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_agent.config import Settings  # noqa: E402
from local_agent.inference.airllm import AirLLMBackend  # noqa: E402
from local_agent.inference.base import InferenceError  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show resolved settings without downloading or loading the model.",
    )
    return parser.parse_args()


def _print_config(settings: Settings, backend: AirLLMBackend) -> None:
    print(f"Model:           {settings.model_id}")
    print(f"Device:          {settings.device}")
    print(f"Maximum context: {settings.max_context}")
    print(f"Hugging Face:    {settings.hf_home}")
    print(f"Layer shards:    {backend.model_shards_path}")
    print(f"Compression:     {settings.compression or 'none'}")
    print(f"HF token set:    {'yes' if settings.hf_token else 'no (public access)'}")


def main() -> int:
    args = _arguments()
    try:
        settings = Settings.from_environment()
        backend = AirLLMBackend(settings)
        _print_config(settings, backend)
        if args.show_config:
            return 0

        print("Preparing model. The first run downloads and splits model weights...")
        asyncio.run(backend.ensure_loaded())
        print(f"Model ready: {backend.model_id}")
    except (InferenceError, OSError, ValueError) as exc:
        print(f"Model preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
