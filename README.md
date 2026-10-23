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

> Paper: *coming soon*. Please cite it, and every dataset you use (see [Citation](#citation)).

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

`datasets.csv` lists every dataset with its category, population, participants, hours, download link and paper.

## Installation

```bash
git clone https://github.com/sposso/fnirs-datasets.git && cd fnirs-datasets
pip install -r requirements.txt
```

## Getting the data

The data are not redistributed. `datasets.csv` gives the download link of every dataset. Download each dataset
into `$FNIRS_DATA_ROOT/<folder>`; see [docs/getting_data.md](docs/getting_data.md) for the few datasets that need
a conversion step first.

## Usage

```python
from loaders import load

records = load("fresh_motor")            # {recording_id: record}
recording_id, record = next(iter(records.items()))
record["data"].shape                     # (channels, 2, samples): [HbO, HbR] in M, 7.5 Hz
```

The dataset names are the keys of `loaders.PRETRAINING` and `loaders.DOWNSTREAM`. Each record holds:

| Key | Content |
|---|---|
| `data` | `(C, 2, T)` array: HbO and HbR concentration changes (M) of `C` channels at 7.5 Hz |
| `channel_xyz` | `(C, 3)` channel positions (source-detector midpoints) in MNI space, meters |
| `sd_distances` | `(C,)` source-detector distances, meters |
| `ch_names` | channel names, e.g. `"S1_D1"` |
| `events` | `[(onset s, name), ...]` |
| `subject_id` | participant identifier |

Downstream records add `epochs` (`(n, C, 2, T)`, one epoch per labeled event) and `labels`. See `examples/`.

## Processing

The same minimal pipeline for every dataset: optical density → HbO/HbR (modified Beer–Lambert law), high-pass
filter at 0.01 Hz, resampling to 7.5 Hz. No channel is rejected for signal quality and no motion correction is
applied. Details, including how each dataset's positions were brought to MNI space, are in
[docs/processing.md](docs/processing.md) and in each loader's docstring.

## Downstream tasks

| Task | Classes | Epoch | Name |
|---|---|---|---|
| ICH | healthy vs. ICH | 10-s motor imagery | `hefmi_ich` |
| CI status | normal hearing vs. cochlear implant | 13-s sentence | `speech_comprehension` |
| Tinnitus | control vs. tinnitus | 12-s noise block | `tinnitus` |
| Movement | movement vs. no movement | 12 s from each LED flash | `stroke_recovery` |
| Action mode | observation vs. execution | 4-s action trial | `mirror` |
| Semantic category | animal vs. tool | 14.3-s trial | `tools_animals` |

## Rejected datasets

24 more datasets were reviewed and not used, mainly for lacking 3D optode positions. Their loaders and the reasons
are in [rejected/REJECTED.md](rejected/REJECTED.md).

## Citation

Please cite our paper (BibTeX coming soon) and each dataset you use; the data record and paper of every dataset are
in `datasets.csv`.

## License

The code is under the MIT License. The datasets keep the licenses of their original repositories.
