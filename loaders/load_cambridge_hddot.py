"""
Butters, E., Collins-Jones, L., Mesquita, R. C., Acharya, D., McKiernan, E., Laurell, A. A. S., Low, A., Srinivasan, S.,
O'Brien, J. T., Su, L., Bale, G. (2026). Brain network analysis in Alzheimer's disease and mild cognitive impairment
using high-density diffuse optical tomography. Imaging Neuroscience 4. doi:10.1162/IMAG.a.1208.
Data: OpenNeuro ds007719 v1.0.0 ("Cambridge_data_resting"; mirrored on NEMAR as on007719), read from the copies made
by scripts/fix_cambridge_hddot.py (SNIRF metadata completed so that MNE can read them; data unchanged).

65 older adults (21 with Alzheimer's disease, 22 with mild cognitive impairment, 22 healthy controls), 5 min of rest
with eyes closed, mostly recorded at home; three files (sub-s02, sub-s03, sub-s05, ~24 min) also hold a task recorded
before the rest and are kept whole. Gowerlabs LUMO, 12 modules over the frontal cortex: 36 sources (735/850 nm) and
48 detectors, 12.5 Hz. The files hold all 1728 source-detector pairs; only pairs up to 35 mm apart are kept
(~108 of them short, < 12 mm): with LUMO, most channels beyond ~35 mm fail basic quality criteria (Vidal-Rosas et al.
2021, Neurophotonics 8:025002), and in this dataset almost none beyond 42.5 mm passed the authors' checks (paper,
Fig. A1).
Positions: digitized on each head by photogrammetry, in mm, with the nasion, inion, preauricular points and Cz
(coordsystem.json); fitted to the fsaverage head with uniform scaling (mne.coreg.Coregistration, all optodes as
head-shape points) and returned in MNI by utils.optode_mni. Beer–Lambert uses the digitized source-detector
distances.
Excluded: sub-s33, sub-s40 and sub-s47, whose landmarks are wrong in the files (e.g. the preauricular points of
sub-s33 are 84 mm apart, on the same side of the head), so that their positions cannot be moved to MNI.
Missing samples: the last 4 s of sub-s75 are NaN in 45 pairs and are cut off by utils.fill_or_drop.
"""

from pathlib import Path

import mne

from .utils import DATA_ROOT, fill_or_drop, optode_mni, print_summary, raw_to_record

MAX_SEPARATION = 0.035  # m, longest source-detector separation kept
EXCLUDED = {"sub-s33", "sub-s40", "sub-s47"}  # wrong landmarks in the files: positions cannot be moved to MNI


def load_cambridge_hddot_data(main_dir=DATA_ROOT / "cambridge_hddot", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*_nirs.snirf")):
        subject = path.name.split("_")[0]                                   # e.g. sub-s04
        if subject in EXCLUDED:
            continue
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        optode_xyz = optode_mni(raw, path.parent / f"{subject}_coordsystem.json")  # fitted with all optodes
        distances = mne.preprocessing.nirs.source_detector_distances(raw.info)
        raw.pick([name for name, d in zip(raw.ch_names, distances) if d <= MAX_SEPARATION])
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw, optode_xyz=optode_xyz)
        record["subject_id"] = subject
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_cambridge_hddot_data()
