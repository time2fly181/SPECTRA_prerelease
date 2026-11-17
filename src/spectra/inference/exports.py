"""Per-epoch annotation exports shared by the CLI, the scoring runtime, and the GUI.

This module has no Qt dependency so the command-line path can write the same
files as the desktop application.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ANNOTATION_FORMATS: tuple[str, ...] = ("csv", "edf", "both")
DEFAULT_ANNOTATION_FORMAT = "csv"
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.65

_STAGE_LABELS = ("W", "N1", "N2", "N3", "REM")
_EDF_STAGE_NAMES = {
    0: "Sleep stage W",
    1: "Sleep stage N1",
    2: "Sleep stage N2",
    3: "Sleep stage N3",
    4: "Sleep stage R",
}
_DEFAULT_START_TIME = datetime(2000, 1, 1, 0, 0, 0)


def format_elapsed_time_hms(time_sec: int) -> str:
    """Format elapsed time in HH:MM:SS."""
    hours = int(time_sec // 3600)
    minutes = int((time_sec % 3600) // 60)
    seconds = int(time_sec % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _stage_label(stage: int) -> str:
    """Map a stage index to its label; ``-1`` is Unscored."""
    if 0 <= stage < len(_STAGE_LABELS):
        return _STAGE_LABELS[stage]
    if stage == -1:
        return "Unscored"
    return str(stage)


def _clamp_window(
    score_window: tuple[int, int] | None, n_epochs: int
) -> tuple[int, int]:
    """Clamp a half-open analysis window to ``[0, n_epochs]``."""
    if score_window is None:
        return 0, n_epochs
    start = max(0, min(int(score_window[0]), n_epochs))
    end = max(start, min(int(score_window[1]), n_epochs))
    return start, end


def _override_epochs(
    manual_overrides: Mapping[int, int] | None, n_epochs: int
) -> set[int]:
    """Return the epoch indices that carry a valid manual stage override."""
    epochs: set[int] = set()
    for raw_epoch, raw_stage in (manual_overrides or {}).items():
        try:
            epoch, stage = int(raw_epoch), int(raw_stage)
        except (TypeError, ValueError):
            continue
        if 0 <= epoch < n_epochs and 0 <= stage < len(_STAGE_LABELS):
            epochs.add(epoch)
    return epochs


def write_epoch_csv(
    path: str | Path,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    *,
    confidences: np.ndarray | None = None,
    epoch_sec: int = 30,
    score_window: tuple[int, int] | None = None,
    model_predictions: np.ndarray | None = None,
    manual_overrides: Mapping[int, int] | None = None,
    threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> int:
    """Write the detailed per-epoch CSV.

    Columns: epoch number (from 1), elapsed time, final stage, model stage,
    manual-override flag, confidence with a low-confidence flag and the
    runner-up stage for flagged epochs, and per-stage probabilities. Epochs
    outside ``score_window`` are reported as Unscored.

    Args:
        path: Destination ``.csv`` path.
        predictions: Final stage indices ``[N]`` (``-1`` for Unscored).
        probabilities: Stage probabilities ``[N, 5]`` or ``None``.
        confidences: Per-epoch confidence ``[N]``; defaults to the max probability.
        epoch_sec: Epoch length in seconds.
        score_window: Half-open analysis window, or ``None`` for all epochs.
        model_predictions: Model stage indices before manual overrides.
        manual_overrides: Mapping of epoch index to overridden stage index.
        threshold: Confidence below which an epoch is flagged.

    Returns:
        Number of epochs written.
    """
    final = np.asarray(predictions, dtype=np.int64)
    n_epochs = len(final)
    model = (
        np.asarray(model_predictions, dtype=np.int64)
        if model_predictions is not None and len(model_predictions) == n_epochs
        else final
    )
    probs = (
        np.asarray(probabilities, dtype=np.float32)
        if probabilities is not None
        else None
    )
    if confidences is not None:
        conf = np.asarray(confidences, dtype=np.float32)
    elif probs is not None:
        conf = np.max(probs, axis=-1).astype(np.float32, copy=False)
    else:
        conf = None
    overridden = _override_epochs(manual_overrides, n_epochs)
    w_start, w_end = _clamp_window(score_window, n_epochs)

    header = [
        "Epoch",
        "Time_sec",
        "Time_hms",
        "Stage",
        "Stage_numeric",
        "ModelStage",
        "ModelStageNumeric",
        "FinalStage",
        "FinalStageNumeric",
        "ManualOverride",
    ]
    if n_epochs and conf is not None:
        header += [
            "Confidence",
            "LowConfidenceFlag",
            "NextMostProbableStage",
            "NextMostProbableProbability",
        ]
    if n_epochs and probs is not None:
        header += ["P_Wake", "P_N1", "P_N2", "P_N3", "P_REM"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for index in range(n_epochs):
            model_stage = int(model[index])
            in_window = w_start <= index < w_end
            final_stage = int(final[index]) if in_window else -1
            final_label = _stage_label(final_stage)
            time_sec = index * epoch_sec
            epoch_probs = (
                probs[index] if probs is not None and index < len(probs) else None
            )
            row: list[object] = [
                index + 1,
                time_sec,
                format_elapsed_time_hms(time_sec),
                final_label,
                final_stage,
                _stage_label(model_stage),
                model_stage,
                final_label,
                final_stage,
                "Yes" if index in overridden else "No",
            ]
            if conf is not None:
                confidence = float(conf[index]) if index < len(conf) else 0.0
                low = confidence < threshold
                next_stage = ""
                next_prob = ""
                if (
                    low
                    and 0 <= model_stage < len(_STAGE_LABELS)
                    and epoch_probs is not None
                    and len(epoch_probs) == len(_STAGE_LABELS)
                ):
                    alt = np.asarray(epoch_probs, dtype=np.float32).copy()
                    alt[model_stage] = -np.inf
                    next_idx = int(np.argmax(alt))
                    if np.isfinite(float(alt[next_idx])):
                        next_stage = _STAGE_LABELS[next_idx]
                        next_prob = f"{float(epoch_probs[next_idx]):.4f}"
                row += [
                    f"{confidence:.4f}",
                    "Yes" if low else "No",
                    next_stage,
                    next_prob,
                ]
            if epoch_probs is not None:
                row += [f"{p:.4f}" for p in epoch_probs]
            writer.writerow(row)

    return n_epochs


def write_edf_annotations(
    path: str | Path,
    predictions: np.ndarray,
    *,
    epoch_sec: int = 30,
    score_window: tuple[int, int] | None = None,
    start_time: datetime | None = None,
) -> int:
    """Write sleep stages as an EDF+ annotations file.

    Consecutive epochs with the same stage are merged into one annotation
    (``Sleep stage W/N1/N2/N3/R``; ``Unscored`` for epochs without usable
    signal). Epochs outside ``score_window`` are left unannotated. The file
    opens alongside the source recording in EDFbrowser and similar viewers.

    Args:
        path: Destination ``.edf`` path.
        predictions: Stage indices ``[N]`` (``-1`` for Unscored).
        epoch_sec: Epoch length in seconds.
        score_window: Half-open analysis window, or ``None`` for all epochs.
        start_time: Recording start; defaults to 2000-01-01 00:00:00.

    Returns:
        Number of annotations written.
    """
    import pyedflib

    stages = np.asarray(predictions, dtype=np.int64)
    w_start, w_end = _clamp_window(score_window, len(stages))

    annotations: list[tuple[float, float, str]] = []
    if w_end > w_start:
        segment_start = w_start
        for i in range(w_start + 1, w_end + 1):
            if i == w_end or stages[i] != stages[segment_start]:
                annotations.append(
                    (
                        segment_start * epoch_sec,
                        (i - segment_start) * epoch_sec,
                        _EDF_STAGE_NAMES.get(int(stages[segment_start]), "Unscored"),
                    )
                )
                segment_start = i

    with pyedflib.EdfWriter(str(path), 0, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setStartdatetime(start_time or _DEFAULT_START_TIME)
        f.setPatientName("Anonymous")
        f.setPatientCode("")
        f.setTechnician("SPECTRA")
        f.setRecordingAdditional("Sleep_staging_annotations")
        for onset, duration, text in annotations:
            f.writeAnnotation(onset, duration, text)

    return len(annotations)


def recording_start_time(edf_path: str | Path) -> datetime | None:
    """Read the recording start time from an EDF header, or ``None`` on failure."""
    from spectra.preprocessing.edf import open_edf_reader

    try:
        with open_edf_reader(str(edf_path)) as (reader, _):
            start = reader.getStartdatetime()
    except Exception as exc:  # header-only best effort; fall back to a fixed date
        logger.warning("Could not read EDF start time from %s: %s", edf_path, exc)
        return None
    return start.replace(tzinfo=None) if isinstance(start, datetime) else None


def export_annotations(
    annotation_format: str,
    output_dir: str | Path,
    edf_basename: str,
    *,
    edf_path: str | Path,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    confidences: np.ndarray | None = None,
    score_window: tuple[int, int] | None = None,
    epoch_sec: int = 30,
) -> dict[str, str]:
    """Write the per-epoch annotation file(s) chosen by ``annotation_format``.

    ``"csv"`` writes ``<basename>_epochs_detailed.csv``, ``"edf"`` writes
    ``<basename>_annotations.edf``, and ``"both"`` writes both. Existing files
    are overwritten.

    Returns:
        Mapping with ``epochs_csv`` and/or ``annotations_edf`` paths.

    Raises:
        ValueError: If ``annotation_format`` is not one of :data:`ANNOTATION_FORMATS`.
    """
    if annotation_format not in ANNOTATION_FORMATS:
        raise ValueError(
            f"annotation_format must be one of {ANNOTATION_FORMATS}, "
            f"got {annotation_format!r}"
        )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if annotation_format in ("csv", "both"):
        csv_path = directory / f"{edf_basename}_epochs_detailed.csv"
        write_epoch_csv(
            csv_path,
            predictions,
            probabilities,
            confidences=confidences,
            epoch_sec=epoch_sec,
            score_window=score_window,
        )
        logger.info("Saved per-epoch CSV to: %s", csv_path)
        paths["epochs_csv"] = str(csv_path)

    if annotation_format in ("edf", "both"):
        edf_out = directory / f"{edf_basename}_annotations.edf"
        write_edf_annotations(
            edf_out,
            predictions,
            epoch_sec=epoch_sec,
            score_window=score_window,
            start_time=recording_start_time(edf_path),
        )
        logger.info("Saved EDF+ annotations to: %s", edf_out)
        paths["annotations_edf"] = str(edf_out)

    return paths
