# Example: score a public recording

This walkthrough scores one publicly available overnight recording with the
released SPECTRA model and summarizes the result. Commands are run from the
repository root after [installation](../README.md#installation). SPECTRA requires
**Python 3.12–3.14**.

## 1. Download the model and the example recording

```bash
uv run python examples/download_example.py
```

This saves two files and checks their SHA-256 checksums:

| File | Saved to | SHA-256 |
| --- | --- | --- |
| SPECTRA model (79 MB) | `checkpoints/spectra_bp84.ckpt` | `caf081ab008a3c9a51182c2ad227516d5cea6eb4d6286a843b49e27507af9052` |
| Example recording SN007 (127 MB) | `data/example/SN007.edf` | `f3859131cc074d26e272e43b9a9dbf00dd6932a79a1956dd6505a9071742919c` |

Both files are also available from the
[v0.1.0-pre release](https://github.com/time2fly181/SPECTRA_prerelease/releases/tag/v0.1.0-pre)
if you prefer to download them by hand. The recording can also be fetched from
PhysioNet, its original host:

```bash
uv run python examples/download_example.py --only edf \
  --edf-url https://physionet.org/files/hmc-sleep-staging/1.1/recordings/SN007.edf
```

## 2. Score the recording

```bash
uv run spectra \
  --edf data/example/SN007.edf \
  --checkpoint checkpoints/spectra_bp84.ckpt \
  --output outputs/example \
  --start-epoch 0 \
  --end-epoch 1034 \
  --device auto
```

The model chooses EEG, EOG, and chin EMG channels from the EDF header
automatically. For SN007 it uses `EEG C3-M2`, `EEG C4-M1`, `EOG E1-M2`,
`EOG E2-M2`, and `EMG chin`.

`--start-epoch` and `--end-epoch` set the analysis interval in 30-second
epochs; the end is exclusive. SN007 is 1034 epochs long (8 h 37 min), it was
already trimmed to lights-off/lights-on by the data provider, and every epoch
has usable signal on all five channels, so the interval `0`–`1034` is the
whole night. For your own recordings, set the interval to lights-off and
lights-on; leaving both options out scores the whole file.

Add `--annotations edf` to write EDF+ annotations instead of the per-epoch
CSV, or `--annotations both` for both files.

A GPU makes scoring faster (about 10 seconds for this night), but the CPU
(`--device cpu`) also works for a single night.

## 3. Read the results

```bash
uv run python examples/read_results.py outputs/example
```

This prints how many epochs and minutes fall in each stage and the model's
mean confidence. The output folder contains:

| File | Contents |
| --- | --- |
| `SN007_predictions.csv` | One row per 30-second epoch: `Epoch` (from 0) and `Stage` (`W`, `N1`, `N2`, `N3`, `REM`, or `Unscored`) |
| `SN007_probabilities.npy` | Array of shape `[epochs, 5]` with stage probabilities in the order W, N1, N2, N3, REM |
| `SN007_flags.npz` | Per-epoch uncertainty scores (`flag_maxprob`, `flag_margin`, `flag_entropy`) |
| `SN007_epoch_signal_valid.npy` | Per-epoch flag for whether usable signal was present |
| `SN007_preprocessing.json` | Channels used, sampling rate, normalization, and analysis interval |
| `SN007_epochs_detailed.csv` | Per-epoch annotations: `Epoch` (from 1), elapsed time, stage, confidence, low-confidence flag, and per-stage probabilities. Written by default (`--annotations csv`) |
| `SN007_annotations.edf` | EDF+ sleep-stage annotations that open alongside `SN007.edf` in EDF viewers. Written with `--annotations edf` or `--annotations both` |

Load them in Python:

```python
import numpy as np
import pandas as pd

stages = pd.read_csv("outputs/example/SN007_predictions.csv")
probabilities = np.load("outputs/example/SN007_probabilities.npy")
```

### Expected result

A correct installation gives this stage distribution for SN007:

| Stage | W | N1 | N2 | N3 | REM |
| --- | --- | --- | --- | --- | --- |
| Epochs predicted by SPECTRA | 65 | 89 | 532 | 126 | 222 |
| Epochs in the expert scoring | 67 | 87 | 474 | 191 | 215 |

Compared with the expert scoring published with the recording, SPECTRA agrees
on 84.8% of epochs (Cohen's κ = 0.78, macro-F1 = 0.82). Results on a CPU can
differ from these counts by a few epochs.

## 4. Review in the desktop application (optional)

```bash
uv run spectra-gui
```

In **Setup**, choose `data/example/SN007.edf` and
`checkpoints/spectra_bp84.ckpt`, then start scoring. **Review** shows the
hypnogram next to the waveforms and highlights uncertain epochs.

## About the example recording

`SN007.edf` is recording SN007 of the Haaglanden Medisch Centrum (HMC) sleep
staging database, version 1.1, published on PhysioNet:
<https://physionet.org/content/hmc-sleep-staging/1.1/>. It is an anonymized
clinical overnight polysomnogram sampled at 256 Hz with eight channels: EEG
F4-M1, C4-M1, O2-M1, and C3-M2; chin EMG; EOG E1-M2 and E2-M2; and ECG. The
expert sleep scoring (`SN007_sleepscoring.edf`) is available from the same
page; it is not included here.

The database is distributed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
The file is redistributed unmodified. If you use it, cite:

- Alvarez-Estevez, D., & Rijsman, R. (2022). Haaglanden Medisch Centrum sleep
  staging database (version 1.1). PhysioNet. RRID:SCR_007345.
  <https://doi.org/10.13026/t79q-fr32>
- Alvarez-Estevez, D., & Rijsman, R. M. (2021). Inter-database validation of a
  deep learning approach for automatic sleep scoring. *PLoS ONE*, 16(8),
  e0256111. <https://doi.org/10.1371/journal.pone.0256111>
- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
  Components of a new research resource for complex physiologic signals.
  *Circulation*, 101(23), e215–e220.

SN007 belongs to SPECTRA's held-out test set. It was not used to train or
select the model.
