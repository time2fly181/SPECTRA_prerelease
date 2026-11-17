# SPECTRA

Automatic sleep staging from EDF polysomnography recordings, with a desktop
application for reviewing and exporting results.

## Model

SPECTRA assigns one of five sleep stages to each 30-second epoch:
**Wake, N1, N2, N3, or REM**.

The model combines a multirate convolutional neural network (CNN) with a context
transformer. Separate CNN branches process EEG, EOG, and chin EMG signals, while
the transformer uses the surrounding epochs to inform each prediction. Each
prediction uses a window of 21 epochs: the target epoch plus ten on either side.

SPECTRA selects up to two EEG channels, two EOG channels, and one EMG channel.
It resamples signals to 128 Hz and normalizes them automatically. Missing
channels and unusable signal are masked during scoring.

## Installation

SPECTRA requires **Python 3.12–3.14** and [uv](https://docs.astral.sh/uv/getting-started/installation/).

Download or clone this repository, then run the following from its root folder:

```bash
uv sync --locked
```

SPECTRA supports CPU inference, NVIDIA GPUs through CUDA, and Apple GPUs through
MPS. A GPU is optional. On Windows, `uv sync` installs the CUDA 12.8 build of
PyTorch from the PyTorch index; on Linux the default build already includes CUDA.
Check with `uv run python -c "import torch; print(torch.cuda.is_available())"`.

Keep the virtual environment out of OneDrive or other synced folders: sync
placeholders can block package upgrades. Set `UV_PROJECT_ENVIRONMENT` to a path
outside the synced folder if the repository lives inside one.

### Model checkpoint

Download `spectra_bp84.ckpt` (79 MB) from the
[v0.1.0-pre release](https://github.com/time2fly181/SPECTRA_prerelease/releases/tag/v0.1.0-pre)
and place it in `checkpoints/`, or run:

```bash
uv run python examples/download_example.py --only checkpoint
```

SHA-256: `caf081ab008a3c9a51182c2ad227516d5cea6eb4d6286a843b49e27507af9052`

## Quick start with the example recording

[examples/README.md](examples/README.md) walks through downloading a public
example recording (night SN007 of the HMC sleep staging database on PhysioNet,
CC BY 4.0), scoring it, and reading the results:

```bash
uv run python examples/download_example.py
uv run spectra --edf data/example/SN007.edf \
  --checkpoint checkpoints/spectra_bp84.ckpt --output outputs/example \
  --start-epoch 0 --end-epoch 1034
uv run python examples/read_results.py outputs/example
```

## Run inference

### Desktop application

```bash
uv run spectra-gui
```

1. In **Setup**, choose your EDF recording and model file. Review the channel
   selections and adjust them if needed. Under **Annotations**, choose which
   per-epoch annotation file to write: **CSV** (default), **EDF+**, or **Both**
   (see [Annotation exports](#annotation-exports)).
2. Start scoring. When scoring finishes, the results and the chosen annotation
   file are saved to the output directory automatically.
3. Use **Review** to inspect the hypnogram and waveforms, review uncertain
   predictions, and make manual corrections. The automatic export contains the
   model's stages only, so after making corrections use **Export** to save an
   updated CSV or EDF+ file.

### Command line

```bash
uv run spectra \
  --edf /path/to/recording.edf \
  --checkpoint checkpoints/spectra_bp84.ckpt \
  --output outputs/recording \
  --annotations csv \
  --device auto
```

Replace the example paths with your files. `--device auto` selects an available
GPU or falls back to the CPU; use `--device cpu` to run on the CPU explicitly.
`--output` defaults to `./outputs`. `--annotations` chooses the per-epoch
annotation file: `csv` (default), `edf`, or `both`.

Results are saved in the output folder, including a CSV of predicted stages,
NumPy arrays of stage probabilities, signal-quality and preprocessing
information, and the annotation file(s). Epochs without usable signal are marked
**Unscored**.

### Annotation exports

Every run, from the command line or the desktop application, writes a
per-epoch annotation file next to the other results. The format is chosen with
`--annotations` or the **Annotations** setting in Setup:

| Choice | File | Contents |
| --- | --- | --- |
| `csv` (default) | `<recording>_epochs_detailed.csv` | One row per 30-second epoch: epoch number (from 1), elapsed time, stage, confidence, a low-confidence flag (confidence below 0.65) with the next most probable stage, and the probability of each stage |
| `edf` | `<recording>_annotations.edf` | EDF+ annotations file with one `Sleep stage W/N1/N2/N3/R` annotation per run of consecutive epochs, timed from the recording's start time; open it alongside the recording in EDFbrowser or other EDF+ software |
| `both` | Both files | |

Epochs outside the analysis interval are marked `Unscored` in the CSV and
left unannotated in the EDF+ file. Existing files with the same name are
overwritten.

For additional options, including the analysis interval, batch size, and
uncertainty sampling:

```bash
uv run spectra --help
```
