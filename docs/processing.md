# Processing

Every dataset goes through the same minimal processing, so that recordings from different devices and laboratories
share one format. The shared code is in `loaders/utils.py`. The docstring of each loader states every
dataset-specific step, with its source.

## Signal

`raw_to_record` processes recordings of raw light intensity:

1. **Optical density.** `mne.preprocessing.nirs.optical_density`. Recordings that are stored as optical density
   start at step 2.
2. **Hemoglobin.** Modified Beer–Lambert law, `mne.preprocessing.nirs.beer_lambert_law`, with a partial pathlength
   factor of 6 (`PPF`). The source-detector distance of each channel comes from the optode positions. When the
   positions are template positions, the template distances are used.
3. **High-pass filter.** 0.01 Hz (`L_FREQ`), 4th-order Butterworth, zero phase.
4. **Resampling.** 7.5 Hz (`TARGET_SFREQ`), `mne.io.Raw.resample`.

Datasets that share only the HbO and HbR computed by the device software, without raw intensity, go through steps
3–4 only (`hb_to_record`).

No channel is rejected for signal quality, and no motion correction or other denoising is applied. The only channels
removed are these:
- **Author flags.** Channels flagged as bad by the dataset authors.
- **Missing samples.** `fill_or_drop` handles NaN and non-positive intensity:
  - A missing start or end of up to 1 min is cut.
  - Gaps of up to 1 s are filled by linear interpolation.
  - A channel that would need more than this is dropped.
- **HD-DOT arrays.** Source-detector pairs longer than 35 mm are dropped.

## Channel positions

`channel_xyz` holds the midpoint of each source-detector pair in MNI space, in meters. Each dataset's positions are
converted with the method that fits what the dataset provides:

| What the dataset provides | Conversion | Function |
|---|---|---|
| MNI coordinates published by the authors | used as given | `raw_to_record(channel_pos=...)` |
| 10-5 labels of the optodes on a template montage | NIRx ICBM152 coordinates of those labels (`nirx_icbm152_1005.json`) | `raw_to_record(optode_xyz=...)` |
| Positions in MNE head (CapTrak) coordinates | `mne.head_to_mni` on fsaverage | `head_to_mni` |
| Digitized positions with nasion and preauricular points | fsaverage fitted with `mne.coreg.Coregistration` (uniform scaling, landmarks, then ICP) | `coreg_mni`, `optode_mni` |
| Template positions on another atlas head with 10-5 landmarks | rigid fit of the landmarks to MNI | `landmark_fit_mni`, `colin27_optode_mni` |

The functions that use fsaverage download it on first use, with `mne.datasets.fetch_fsaverage`.

Datasets with 2D positions only, or with positions in an undocumented frame, are not part of the corpus. See
[`rejected/REJECTED.md`](../rejected/REJECTED.md).

## Downstream epochs

The downstream loaders (`loaders/downstream/`) also return epochs and labels. Event onsets come from the event
files or stimulus markers of each dataset. `epochs()` then cuts one epoch per labeled event with `mne.Epochs`,
covering the entire condition (`CONDITION = (start, duration)` in each loader). No baseline correction,
detrending or amplitude-based rejection is applied. Epochs that extend past the end of a recording are dropped.
