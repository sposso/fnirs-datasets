"""
Anderson, J. E., Carlton, L. B., Kura, S., O'Brien, W. J., Rogers, D., Rahimi, P., Farzam, P. Y., Zaman, M. H.,
Boas, D. A., Yücel, M. A. (2025). High-density multidistance fNIRS enhances detection of brain activity during a
word-color Stroop task. Neurophotonics 12(3), 035010. doi:10.1117/1.NPh.12.3.035010.
Data: OpenNeuro ds006460 (high-density array, v1.0.0) and ds006459 (sparse array, v1.0.1).

17 adults performed a word-color Stroop task twice in one session (~10 min each: 18 blocks of 18 s, 12 blocks for
sub-1 and sub-2), once with each array, in random order. NIRSport2 (760/850 nm), prefrontal cortex and temples:
- high-density ("hd"): hexagonal pattern of 25 sources and 66 detectors, 112 channels at ~19 mm, 94 at ~33 mm and
  8 short ones (~8 mm), 17.44 Hz;
- sparse: 30-mm grid of 17 sources and 24 detectors, 52 channels plus 8 short ones (~8 mm), 24.41 Hz.
All channels are kept, including the short ones. A participant has the same subject_id in both arrays.
Optode positions: the arrays on the NIRx ICBM152 head (the files' 300 10-5 landmarks equal the NIRx ICBM152 table),
i.e. MNI coordinates, identical for every participant; kept as stored.
Missing samples: 6 zero samples in two high-density files (sub-13, sub-17), filled with utils.fill_or_drop.
"""

from pathlib import Path

import mne

from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

ARRAYS = {"hd": DATA_ROOT / "stroop_hd", "sparse": DATA_ROOT / "stroop_sparse"}  # array -> data folder


def load_stroop_data(arrays=ARRAYS, verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file of both arrays."""
    outputs = {}
    for array, main_dir in arrays.items():
        for path in sorted(Path(main_dir).glob("sub-*/ses-1/nirs/*.snirf")):
            raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
            sfreq = raw.info["sfreq"]
            raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
            record = raw_to_record(raw)
            record["subject_id"] = path.name.split("_")[0]  # sub-1 ... sub-17, the same person in both arrays
            outputs[f"{array}_{path.stem}"] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_stroop_data()
