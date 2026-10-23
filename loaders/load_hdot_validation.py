"""
Fogarty, M., Rafferty, S. M., Markow, Z. E., O'Sullivan, A. C., Svoboda, C. F., George, T., King, K., Wilhelm, D.,
Tripathy, K., Mugler, E. M., Naufel, S., Yin, A., Trobaugh, J. W., Eggebrecht, A. T., Richter, E. J., Culver, J. P.
(2025). Functional brain mapping using whole-head very high-density diffuse optical tomography. Imaging Neuroscience 3,
IMAG.a.54. doi:10.1162/IMAG.a.54.
Data: OpenNeuro ds007463 v1.1.1 ("Very-High-Density Diffuse Optical Tomography System Validation Dataset"; its fNIRS
files are unchanged in v2.0.0, which only removed MRI files). The fMRI sessions are not used.

8 adults, two VHD-DOT sessions each, 88 runs of 3-11 min (10.15 h): left/right checkerboard wedges (AC), verb
generation (GV), word hearing (HW), left/right finger tapping (MOT) and a 10-min clip of The Good, the Bad, and the Ugly
watched twice per session (MOV). The paper analyzed 86 of the runs; all are kept.
Custom VHD-DOT system (Washington University in St. Louis): 255 laser-diode sources (685/830 nm) and 252 avalanche
photodiode detectors on a 9.75-mm grid over four cap panels (left, right, posterior, dorsal), 7.81 Hz.
The files hold the 9543-10759 pairs up to 60 mm apart; only pairs up to 35 mm are kept (3325-3652 per run, ~907 of
them short, < 12 mm), as for the other HD-DOT datasets: the first three neighbor rings and part of the fourth (34-38 mm).
Positions: the optodes were placed on each participant's MRI head model (guided by session photographs and
word-hearing fMRI maps) and are the same for both sessions of a participant, in mm, with the nasion, inion and
preauricular points (coordsystem.json); fitted to the fsaverage head with uniform scaling and returned in MNI by
utils.optode_mni.
Missing samples: isolated samples are stored as machine epsilon (2.2e-16) instead of an intensity; they are set to zero
so that utils.fill_or_drop treats them as missing.
MNE cannot read the landmark labels (stored as a 1 x 4 array), so each file is read with utils.read_snirf_flat_labels.
"""

from pathlib import Path

import mne
import numpy as np

from .utils import DATA_ROOT, fill_or_drop, optode_mni, print_summary, raw_to_record, read_snirf_flat_labels

MAX_SEPARATION = 0.035  # m, longest source-detector separation kept


def load_hdot_validation_data(main_dir=DATA_ROOT / "hdot_validation", verbose=True):
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
    load_hdot_validation_data()
