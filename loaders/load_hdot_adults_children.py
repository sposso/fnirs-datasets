"""
Tripathy, K., Fogarty, M., Svoboda, A. M., Schroeder, M. L., Rafferty, S. M., Richter, E. J., Tracy, C., Mansfield,
P. K., Booth, M., Fishell, A. K., Sherafati, A., Markow, Z. E., Wheelock, M. D., Arbeláez, A. M., Schlaggar, B. L.,
Smyser, C. D., Eggebrecht, A. T., Culver, J. P. (2024). Mapping brain function in adults and young children during
naturalistic viewing with high-density diffuse optical tomography. Human Brain Mapping 45(7), e26684.
doi:10.1002/hbm.26684.
Data: OpenNeuro ds007473 v1.0.0 ("High-Density Diffuse Optical Tomography Audiovisual Movie Viewing Dataset").

5 adults (the children's data are not public), 1-8 sessions each, 189 runs of 3-7 min: rotating checkerboard (CCW),
word hearing (HW, and HWM in the movie sessions), finger tapping (MOT) and animated movie clips (MOV, 13.7 of 17.3 h).
Custom HD-DOT system (Washington University in St. Louis): 128 laser-diode sources (685/830 nm) and 125 avalanche
photodiode detectors on an 11-mm grid over four cap panels (occipital, left and right temporal, dorsal), 10.42 Hz.
The files hold the 3375-3515 pairs up to 60 mm apart; only pairs up to 35 mm are kept (~1450 per run, ~427 of them
short, < 12 mm), as for the Cambridge HD-DOT dataset.
Positions: the optodes were placed on each participant's MRI head model (guided by session photographs and auditory
localizer maps) and are the same for all sessions of a participant, in mm, with the nasion, inion and preauricular
points (coordsystem.json); fitted to the fsaverage head with uniform scaling and returned in MNI by utils.optode_mni,
as for the Cambridge dataset.
Missing samples: a few isolated samples are stored as machine epsilon (2.2e-16) instead of an intensity (the other
samples are above 1e-10); they are set to zero so that utils.fill_or_drop treats them as missing.
MNE cannot read the landmark labels (stored as a 1 x 4 array), so each file is read with
utils.read_snirf_flat_labels (a temporary copy with the labels flattened; fixed copies would take 37 GB).
"""

from pathlib import Path

import mne
import numpy as np

from .utils import DATA_ROOT, fill_or_drop, optode_mni, print_summary, raw_to_record, read_snirf_flat_labels

MAX_SEPARATION = 0.035  # m, longest source-detector separation kept


def load_hdot_adults_children_data(main_dir=DATA_ROOT / "hdot_adults_children", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*_nirs.snirf")):
        raw = read_snirf_flat_labels(path)
        subject, session = path.name.split("_")[:2]                         # e.g. sub-A01, ses-1
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
    load_hdot_adults_children_data()
