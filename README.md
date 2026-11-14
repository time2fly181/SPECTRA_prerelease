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
MPS. A GPU is optional.

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
   selections and adjust them if needed.
2. Start scoring.
3. Use **Review** to inspect the hypnogram and waveforms, review uncertain
   predictions, and make manual corrections. Use **Export** to save your results.

### Command line

```bash
uv run spectra \
  --edf /path/to/recording.edf \
  --checkpoint checkpoints/spectra_bp84.ckpt \
  --output outputs/recording \
  --device auto
```

Replace the example paths with your files. `--device auto` selects an available
GPU or falls back to the CPU; use `--device cpu` to run on the CPU explicitly.

Results are saved in the output folder, including a CSV of predicted stages,
NumPy arrays of stage probabilities, and signal-quality and preprocessing
information. Epochs without usable signal are marked **Unscored**.

For additional options, including the analysis interval, batch size, and
uncertainty sampling:

```bash
uv run spectra --help
```
