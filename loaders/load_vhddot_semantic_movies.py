"""
Fehner, W., Fogarty, M., Tang, J., Wilhelm, D., Bajracharya, A., Markow, Z. E., Hines, A., Trobaugh, J. W., Huth, A. G.,
Culver, J. P. (2025). Visual semantic encoding and identification of naturalistic movies via high-density diffuse
optical tomography. bioRxiv. doi:10.64898/2025.12.03.692158.
Data: OpenNeuro ds007816 v1.1.1 (same title; CC0). Only the fNIRS files were downloaded (no MRI or derivatives).

6 adults, 3 same-day sessions each, 178 runs: silent natural movies (12 training runs and 9 repeated-test runs of
10 min per participant, Trn and Val), plus checkerboard (AC) and word-hearing (HW) localizers. No participants or
data were excluded by the authors.
Custom VHD-DOT system (Washington University in St. Louis), the same as load_hdot_validation: 255 laser-diode sources
(685/830 nm) and 252 avalanche photodiode detectors on a 9.75-mm grid, 7.81 Hz. Only pairs up to 35 mm are kept, as
for the other HD-DOT datasets.
Positions: the optodes were placed on each participant's MRI head model (aligned to session photographs, then refined
with the localizer maps) and are identical across the sessions of a participant, in mm, with the nasion, inion and
preauricular points (coordsystem.json); fitted to fsaverage and returned in MNI by utils.optode_mni.
Missing samples: isolated samples stored as machine epsilon (2.2e-16) instead of an intensity are set to zero so that
utils.fill_or_drop treats them as missing.
MNE cannot read the landmark labels (stored as a 1 x 4 array), so each file is read with utils.read_snirf_flat_labels.
"""

from pathlib import Path

import mne
import numpy as np

from .utils import DATA_ROOT, fill_or_drop, optode_mni, print_summary, raw_to_record, read_snirf_flat_labels

MAX_SEPARATION = 0.035  # m, longest source-detector separation kept


def load_vhddot_semantic_movies_data(main_dir=DATA_ROOT / "vhddot_semantic_movies", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*_nirs.snirf")):
        raw = read_snirf_flat_labels(path)
        subject, session = path.name.split("_")[:2]                         # e.g. sub-S01, ses-1
        optode_xyz = optode_mni(raw, path.parent / f"{subject}_{session}_coordsystem.json")  # fitted with all optodes
        distances = mne.preprocessing.nirs.source_detector_distances(raw.info)
        raw.pick([name for name, d in zip(raw.ch_names, distances) if d <= MAX_SEPARATION])
        raw.apply_function(lambda x: np.where(x > np.finfo(float).eps, x, 0.0), channel_wise=False)  # eps: missing
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw, optode_xyz=optode_xyz)
        record["subject_id"] = subject
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_vhddot_semantic_movies_data()
