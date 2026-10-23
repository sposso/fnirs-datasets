# fnirs-datasets

Loaders for 67 public fNIRS datasets in one common format with channel positions in MNI space: 1,954.84 hours of
recordings from 2,798 participants. Each recording becomes HbO and HbR concentration changes at 7.5 Hz, with the MNI
position of every channel. Of the 67, 61 datasets form a pretraining corpus and 6 are held out as downstream tasks
with labeled epochs. The repository gives a download link for every dataset.

The goal is to make foundation models for fNIRS easier to build: pretraining needs many datasets pooled together,
and pooling fNIRS datasets needs a common format and a common spatial reference for channels. This repository
provides both, together with a fixed benchmark to compare models on.

- **Pretraining corpus:** 61 datasets, 2,577 participants, 1,835.88 hours.
- **Downstream tasks:** 6 held-out datasets, 221 participants, 118.96 hours, with labeled epochs.

This repository holds code only. The data stay with their original repositories, under their own licenses;
`datasets.csv` and [Getting the data](#getting-the-data) give the download link of every dataset.

> Paper: *coming soon*. Please cite it, and every dataset you use (see [Citation](#citation)).

## Contents

```
loaders/              one loader per dataset, plus the shared processing (utils.py)
loaders/downstream/   the 6 downstream datasets: records plus epochs and labels
scripts/              one-off conversions and fixes, run once after downloading some datasets
rejected/             loaders of 24 reviewed datasets that are not used, and the reasons (REJECTED.md)
examples/             minimal usage
docs/processing.md    processing, channel positions and epochs in detail
datasets.csv          every dataset: category, participants, hours, folder, source, paper
```

## Corpus overview

| Category | Datasets | Participants | Hours |
|---|---:|---:|---:|
| Social interaction | 7 | 514 | 486.55 |
| Motor | 15 | 381 | 416.80 |
| Cognitive | 14 | 491 | 363.81 |
| Speech | 11 | 644 | 310.62 |
| Resting state | 6 | 353 | 132.61 |
| Visual | 5 | 92 | 67.03 |
| Emotion | 3 | 102 | 58.46 |
| **Total** | **61** | **2,577** | **1,835.88** |

`datasets.csv` lists the subcategory, population, source and paper of each dataset.

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/sposso/fnirs-datasets.git && cd fnirs-datasets
pip install -r requirements.txt
```

Three packages are needed only for specific parts (`mne-nirs` for the conversion scripts, `mat-io` for the
simulated flight dataset, `rarfile` for two rejected datasets); they are included in `requirements.txt`.

## Getting the data

1. Choose a data folder and set `FNIRS_DATA_ROOT` to it. Without it, the loaders look in `./data`.
   ```bash
   export FNIRS_DATA_ROOT=/path/to/fnirs_data
   ```
2. For each dataset you need, download it from `data_source` in `datasets.csv` into `$FNIRS_DATA_ROOT/<folder>`.
   Each loader's docstring says which files it reads inside that folder.
3. A few datasets need a one-off conversion or fix before they can be loaded. The original files are never modified.

| Dataset | Download to | Then run | Why |
|---|---|---|---|
| Resting state in AD and MCI | `cambridge_hddot_download/` | `python -m scripts.fix_cambridge_hddot resting` | SNIRF fields missing, so MNE cannot read the files |
| Cambridge visual task | `cambridge_visual_download/` | `python -m scripts.fix_cambridge_hddot visual` | same |
| Driving | `driving/` | `python -m scripts.recover_driving` | 25 truncated files |
| Hair and skin characteristics | `inclusion/` | `python -m scripts.fix_inclusion_sub400` | two unreadable files |
| Parent–child hyperscanning | `parents_download/` | `python -m scripts.convert_parents` | NIRStar files → SNIRF |
| Stroke BCI | `stroke_bci_download/` | `python -m scripts.fix_stroke_bci` | empty landmark entries |
| Motor recovery after stroke | `stroke_recovery_download/` | `python -m scripts.fix_stroke_recovery` | optical-density label, transposed data, three detector positions |
| Intracerebral hemorrhage | `hefmi_ich_download/source_data/` | `python -m scripts.convert_hefmi_ich` | Homer `.nirs` → SNIRF, with stimulus markers |

Some datasets need extra steps to obtain:
- **Resident physician workload:** the Zenodo files are restricted; request access.
- **Finger tapping:** the data come from a private figshare link given in the paper.
- **Language production:** download the zip from NITRC and keep it as a zip.

## Usage

Load a dataset by name. The names are the keys of `loaders.PRETRAINING` and `loaders.DOWNSTREAM`.

```python
from loaders import load

records = load("fresh_motor")            # {recording_id: record}
recording_id, record = next(iter(records.items()))
record["data"].shape                     # (channels, 2, samples): [HbO, HbR] in M, 7.5 Hz
```

Each loader also runs from the command line and prints a summary:

```bash
python -m loaders.load_fresh_motor
```

### Record format

| Key | Content |
|---|---|
| `data` | `(C, 2, T)` array: HbO and HbR concentration changes (M) of `C` channels at `sfreq` |
| `sfreq` | 7.5 Hz |
| `channel_xyz` | `(C, 3)` channel positions (source-detector midpoints) in MNI space, meters |
| `sd_distances` | `(C,)` source-detector distances, meters |
| `ch_names` | channel names, e.g. `"S1_D1"` |
| `n_channels_original` | number of signals in the original file |
| `events` | `[(onset s, name), ...]`, on the time axis of `data` |
| `subject_id` | participant identifier |

The downstream loaders add these keys:

| Key | Content |
|---|---|
| `event_labels` | `{event name: class}` for the events used by the task |
| `group` | participant group, for the tasks labeled by group |
| `epochs` | `(n, C, 2, T)` array, one epoch per labeled event |
| `labels` | the class of each epoch |

The number of channels and the montage differ between datasets, and sometimes between participants. Use
`channel_xyz` to relate channels across recordings.

## Processing

The processing is minimal and the same for every dataset:

1. Raw intensity → optical density → HbO/HbR (modified Beer–Lambert law, partial pathlength factor 6).
2. High-pass filter at 0.01 Hz (4th-order Butterworth, zero phase).
3. Resampling to 7.5 Hz.

No channel is rejected for signal quality, and no motion correction is applied. The only channels removed are:
- channels flagged as bad by the dataset authors;
- channels with missing samples that cannot be repaired (gaps are filled up to 1 s; a missing start or end is cut up to 1 min);
- HD-DOT source-detector pairs longer than 35 mm.

Channel positions are converted to MNI with the method that fits each dataset: published MNI coordinates,
template 10-5 positions, CapTrak → MNI, or coregistration of digitized positions to fsaverage. See
[docs/processing.md](docs/processing.md). Each loader's docstring records the dataset-specific decisions and their
sources.

## Downstream tasks

| Task | Classes | Epoch | Name |
|---|---|---|---|
| ICH | healthy vs. ICH (participant group) | 10-s motor imagery, 2–12 s after the visual cue | `hefmi_ich` |
| CI status | normal hearing vs. cochlear implant (participant group) | 13-s sentence stimulus | `speech_comprehension` |
| Tinnitus | control vs. tinnitus (participant group) | 12-s broadband noise block | `tinnitus` |
| Movement | movement vs. no movement | 0–12 s from each LED flash | `stroke_recovery` |
| Action mode | observation vs. execution | 4-s action trial | `mirror` |
| Semantic category | animal vs. tool | 14.3-s image, mask and four mental tasks | `tools_animals` |

The labels follow the task definitions of GeoNIRS. Each epoch covers the entire condition, as described in the
dataset's publication. See `examples/downstream_epochs.py`.

## Rejected datasets

24 datasets were reviewed and not used, mainly because they lack 3D optode positions. Their loaders and the reasons
are in [rejected/REJECTED.md](rejected/REJECTED.md).

## Citation

If you use this code, please cite our paper (BibTeX coming soon). Please also cite each dataset you use: its
`data_source` and `paper` are in `datasets.csv`, and its full reference is in the loader's docstring.

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The datasets are not redistributed here. Each
one keeps the license of its original repository.
