"""
Visual task of the Cambridge "Optical Neuroimaging and Cognition" study, the same study and participants as the
resting-state data (loaders/load_cambridge_hddot.py): Butters, E. et al. (2026). Brain network analysis in Alzheimer's
disease and mild cognitive impairment using high-density diffuse optical tomography. Imaging Neuroscience 4.
doi:10.1162/IMAG.a.1208 (which mentions the task paradigms; no paper describes the visual task itself).
Data: OpenNeuro ds007714 v1.0.0 ("Cambridge_data"), read from the copies made by scripts/fix_cambridge_hddot.py
(SNIRF metadata completed so that MNE can read them; data unchanged).

64 older adults, one visual-task run each (8-11 min). Gowerlabs LUMO, 6 modules over the back of the head:
18 sources (735/850 nm) and 24 detectors, 432 pairs, 5 Hz. Only pairs up to 35 mm apart are kept, as for the
resting-state data.
Positions: digitized on each head, in mm, with the nasion, inion, preauricular points and Cz (coordsystem.json);
returned in MNI by utils.optode_mni, as for the resting-state data.
Skipped: sub-s45, whose file is truncated in the upload and cannot be opened (not copied by the fix script).
"""

from pathlib import Path

import mne

from .utils import DATA_ROOT, fill_or_drop, optode_mni, print_summary, raw_to_record

MAX_SEPARATION = 0.035  # m, longest source-detector separation kept


def load_cambridge_visual_data(main_dir=DATA_ROOT / "cambridge_visual", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*_nirs.snirf")):
        subject = path.name.split("_")[0]                                   # e.g. sub-s04
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
    load_cambridge_visual_data()
