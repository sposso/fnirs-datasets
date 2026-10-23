"""
Bergonzi, K. M., Burke, B. A., Fogarty, M., Sherafati, A., Burns-Yocum, T. M., Ferradal, S. L., Palanca, B. J. A.,
Dhar, R., Kumar, G., Lee, J.-M., Culver, J. P., Eggebrecht, A. T. (2026). Disrupted functional connectivity during
stroke recovery revealed via bedside optical neuroimaging. Neurophotonics 13(3), 035002. doi:10.1117/1.NPh.13.3.035002.
Data: OpenNeuro ds008798 v2.1.0 (raw SNIRF; the reconstructed NIfTI derivatives are not used).

13 patients with acute ischemic (MCA) stroke, resting state at the bedside in the ICU within 72 h of onset:
54 runs of 3-11 min (8.24 h), 1-9 per patient. Clinical HD-DOT system (Washington University in St. Louis): 48 LED
sources (750/850 nm) and 34 avalanche photodiode detectors in two arrays over both hemispheres, first-to-fourth
neighbor separations of 13, 30, 39 and 47 mm, 10.0 Hz. Only pairs up to 35 mm are kept, as for the other HD-DOT
datasets (the first and second neighbors).
Positions: the arrays on the MNI152 atlas head, adjusted for each patient's cap position and head size (paper), in mm,
with the nasion, inion and preauricular points (coordsystem.json); fitted to the fsaverage head with uniform scaling
and returned in MNI by utils.optode_mni, as for the other HD-DOT datasets.
MNE cannot read the landmark labels (stored as a 1 x 4 array), so each file is read with utils.read_snirf_flat_labels.
"""

from pathlib import Path

import mne
import numpy as np

from .utils import DATA_ROOT, fill_or_drop, optode_mni, print_summary, raw_to_record, read_snirf_flat_labels

MAX_SEPARATION = 0.035  # m, longest source-detector separation kept


def load_stroke_hddot_data(main_dir=DATA_ROOT / "stroke_hddot", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*_nirs.snirf")):
        raw = read_snirf_flat_labels(path)
        subject, session = path.name.split("_")[:2]                         # e.g. sub-P01, ses-1
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
    load_stroke_hddot_data()
