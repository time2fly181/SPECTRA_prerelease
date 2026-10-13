"""Summarize a SPECTRA output folder.

Prints the number of epochs, minutes per stage, and mean model confidence
from the ``*_predictions.csv`` and ``*_probabilities.npy`` files that
``spectra`` writes.

Usage::

    uv run python examples/read_results.py outputs/example
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

STAGES = ["W", "N1", "N2", "N3", "REM"]
EPOCH_SECONDS = 30


def summarize(output_dir: Path) -> None:
    """Print a per-stage summary for every recording in ``output_dir``.

    Args:
        output_dir: Folder passed to ``spectra --output``.

    Raises:
        ValueError: If the folder contains no prediction files.
    """
    csv_paths = sorted(output_dir.glob("*_predictions.csv"))
    if not csv_paths:
        raise ValueError(f"No *_predictions.csv files found in {output_dir}.")

    for csv_path in csv_paths:
        recording = csv_path.name.removesuffix("_predictions.csv")
        with csv_path.open(newline="") as f:
            stages = [row["Stage"] for row in csv.DictReader(f)]
        probabilities = np.load(output_dir / f"{recording}_probabilities.npy")

        scored = np.array([stage != "Unscored" for stage in stages])
        confidence = probabilities[scored].max(axis=1) if scored.any() else None

        print(f"\n{recording}: {len(stages)} epochs, {int(scored.sum())} scored")
        print(f"  {'Stage':<9}{'Epochs':>7}{'Minutes':>9}{'Percent':>9}")
        for stage in [*STAGES, "Unscored"]:
            count = stages.count(stage)
            if stage == "Unscored" and count == 0:
                continue
            minutes = count * EPOCH_SECONDS / 60
            percent = 100 * count / len(stages)
            print(f"  {stage:<9}{count:>7}{minutes:>9.1f}{percent:>8.1f}%")
        if confidence is not None:
            print(f"  Mean confidence (max probability): {confidence.mean():.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path, help="Folder written by spectra")
    args = parser.parse_args()
    try:
        summarize(args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
