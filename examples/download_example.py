"""Download the SPECTRA checkpoint and the example recording.

Files are saved under ``checkpoints/`` and ``data/example/`` in the repository
root and verified against their published SHA-256 checksums. Files that are
already present and pass the checksum are not downloaded again.

Usage::

    uv run python examples/download_example.py
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_URL = (
    "https://github.com/time2fly181/SPECTRA_prerelase/releases/download/v0.1.0-pre"
)


@dataclass(frozen=True)
class Asset:
    """A downloadable file with its destination and checksum."""

    name: str
    url: str
    sha256: str
    destination: Path


CHECKPOINT = Asset(
    name="checkpoint",
    url=f"{RELEASE_URL}/spectra_bp84.ckpt",
    sha256="caf081ab008a3c9a51182c2ad227516d5cea6eb4d6286a843b49e27507af9052",
    destination=REPO_ROOT / "checkpoints" / "spectra_bp84.ckpt",
)

# Recording SN007 of the Haaglanden Medisch Centrum sleep staging database
# (PhysioNet, CC BY 4.0). The same file is published by PhysioNet at
# PHYSIONET_EDF_URL; pass it with --edf-url to download from the original host.
PHYSIONET_EDF_URL = (
    "https://physionet.org/files/hmc-sleep-staging/1.1/recordings/SN007.edf"
)
EXAMPLE_EDF = Asset(
    name="example recording",
    url=f"{RELEASE_URL}/SN007.edf",
    sha256="f3859131cc074d26e272e43b9a9dbf00dd6932a79a1956dd6505a9071742919c",
    destination=REPO_ROOT / "data" / "example" / "SN007.edf",
)


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(asset: Asset, url: str | None = None) -> None:
    """Download ``asset`` unless a verified copy already exists.

    Args:
        asset: File to fetch.
        url: Optional replacement for ``asset.url`` (for example, a mirror).

    Raises:
        ValueError: If the asset has no URL or the checksum does not match.
    """
    source = url or asset.url
    if not source:
        raise ValueError(f"No download URL is configured for the {asset.name}.")
    if asset.destination.exists() and sha256_of(asset.destination) == asset.sha256:
        print(f"{asset.name}: already present at {asset.destination}")
        return

    asset.destination.parent.mkdir(parents=True, exist_ok=True)
    partial = asset.destination.with_name(asset.destination.name + ".part")
    print(f"{asset.name}: downloading {source}")
    with urllib.request.urlopen(source) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)

    actual = sha256_of(partial)
    if actual != asset.sha256:
        partial.unlink()
        raise ValueError(
            f"{asset.name}: checksum mismatch (expected {asset.sha256}, got {actual})."
        )
    partial.replace(asset.destination)
    print(f"{asset.name}: saved to {asset.destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint-url", help="Alternative checkpoint URL")
    parser.add_argument("--edf-url", help="Alternative example-recording URL")
    parser.add_argument(
        "--only",
        choices=["checkpoint", "edf"],
        help="Download only one of the two files",
    )
    args = parser.parse_args()

    try:
        if args.only in (None, "checkpoint"):
            fetch(CHECKPOINT, args.checkpoint_url)
        if args.only in (None, "edf"):
            fetch(EXAMPLE_EDF, args.edf_url)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
