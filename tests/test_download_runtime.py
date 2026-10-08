"""Safety tests for the llama.cpp runtime installer."""

import zipfile
from pathlib import Path

import pytest

from scripts.download_runtime import safe_extract


def test_safe_extract_accepts_normal_archive(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.zip"
    destination = tmp_path / "runtime"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("llama-server.exe", b"test")

    safe_extract(archive, destination)

    assert (destination / "llama-server.exe").read_bytes() == b"test"


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", b"unsafe")

    with pytest.raises(RuntimeError, match="Unsafe archive member"):
        safe_extract(archive, tmp_path / "runtime")
