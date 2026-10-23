"""
Rocco, G., Chalet, L., Fear, E. J., Pomante, S., Graziano, F., Di Censo, D., Carriero, M., Delaire, E., Esposito, F.,
Perrucci, M. G., Del Gratta, C., Perpetuini, D., Wise, R. G., Chiarelli, A. M. (2026). A comparison of continuous-wave
fNIRS with quantitative fMRI-derived indices of brain function in the visual cortex. bioRxiv 2026.08.10.743546.
doi:10.64898/2026.08.10.743546.
Data: "DOMRIBrain" (CC0), G-Node GIN repository giuliarocco/DOMRIBrain (master, fNIRS files only, downloaded
2026-10-16; the SNIRF files match the repository's git-annex checksums).

13 healthy adults, one session inside a 3 T MRI scanner, two runs of ~11 min: a 2-Hz flickering checkerboard and a
hypercapnic breath-hold task. NIRx Borealis (MR-compatible laser sources, 785/830 nm, avalanche photodiodes), 14.8 Hz,
EasyCap 10-5 cap over the occipital lobe: 54 long (21-42 mm) and 8 short (8 mm) channels. The files list all 114
source-detector pairs; only the 62 pairs of the montage (up to 45 mm) are kept.
Excluded: sub-03, sub-09 and sub-10, excluded by the authors for insufficient fNIRS quality (README: fewer than 60% of
the channels passing their SCI/PSP check).
Positions: the SNIRF files store the NIRx ICBM152 template (their 300 10-5 landmarks equal the NIRx ICBM152 table),
i.e. MNI coordinates, the same for every participant; kept as stored.
Missing samples: a few isolated zero or negative intensity samples, filled by utils.fill_or_drop.
"""

from pathlib import Path

import mne

from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

EXCLUDED = {"sub-03", "sub-09", "sub-10"}  # excluded by the authors (README): < 60% of channels pass their QC
MAX_SEPARATION = 0.045  # m: the montage; the other pairs listed in the files are not channels


def load_domribrain_data(main_dir=DATA_ROOT / "domribrain", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*_nirs.snirf")):
        subject = path.name.split("_")[0]                                   # e.g. sub-01
        if subject in EXCLUDED:
            continue
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        distances = mne.preprocessing.nirs.source_detector_distances(raw.info)
        raw.pick([name for name, d in zip(raw.ch_names, distances) if d <= MAX_SEPARATION])
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw)
        record["subject_id"] = subject
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_domribrain_data()
