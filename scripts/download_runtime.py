"""Install the tested Windows CUDA build of llama.cpp into .runtime."""

from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / ".runtime" / "llama.cpp"
DOWNLOAD_DIR = PROJECT_ROOT / ".runtime" / "downloads"
CHUNK_SIZE = 1024 * 1024
REPORT_EVERY = 64 * CHUNK_SIZE


@dataclass(frozen=True, slots=True)
class Asset:
    name: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://github.com/ggml-org/llama.cpp/releases/download/b11503/{self.name}"


ASSETS = (
    Asset(
        name="llama-b11503-bin-win-cuda-12.4-x64.zip",
        size=265_086_606,
        sha256="d422d09acc9a177eda24df36fe2ffefdf161214d8ce6002031d50796de383404",
    ),
    Asset(
        name="cudart-llama-bin-win-cuda-12.4-x64.zip",
        size=391_443_627,
        sha256="8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, asset: Asset) -> None:
    if path.stat().st_size != asset.size:
        raise RuntimeError(
            f"Unexpected size for {asset.name}: {path.stat().st_size}; "
            f"expected {asset.size}."
        )
    actual_hash = file_sha256(path)
    if actual_hash != asset.sha256:
        raise RuntimeError(
            f"Unexpected SHA-256 for {asset.name}: {actual_hash}; "
            f"expected {asset.sha256}."
        )


def download(asset: Asset) -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = DOWNLOAD_DIR / asset.name
    if destination.exists():
        verify(destination, asset)
        print(f"Archive already verified: {asset.name}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(
        asset.url, headers={"User-Agent": "RepoWren-setup"}
    )
    if existing:
        request.add_header("Range", f"bytes={existing}-")

    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", 200)
        if existing and status != 206:
            print(f"Restarting {asset.name}; the server did not honor resume.")
            existing = 0
            mode = "wb"
        else:
            mode = "ab" if existing else "wb"

        downloaded = existing
        next_report = downloaded + REPORT_EVERY
        with partial.open(mode) as output:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(
                        f"{asset.name}: {downloaded / asset.size * 100:.1f}%",
                        flush=True,
                    )
                    next_report += REPORT_EVERY

    verify(partial, asset)
    partial.replace(destination)
    return destination


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_root):
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def main() -> int:
    try:
        for asset in ASSETS:
            archive = download(asset)
            print(f"Extracting {asset.name}")
            safe_extract(archive, RUNTIME_DIR)

        server = RUNTIME_DIR / "llama-server.exe"
        if not server.is_file():
            raise RuntimeError(f"llama-server.exe was not found in {RUNTIME_DIR}")
        print(f"llama.cpp ready: {server}")
    except (OSError, RuntimeError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        print(f"Runtime installation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
