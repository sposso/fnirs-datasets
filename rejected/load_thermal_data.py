"""
Yücel, M. A., Aasted, C. M., Petkov, M. P., Borsook, D., Boas, D. A., Becerra, L. (2015). Specific somatosensory
cortex responses to painful and non-painful stimuli: an fNIRS study. Sci. Rep. 5, 9469.
Yücel, M. A., Selb, J., Aasted, C. M., Petkov, M. P., Becerra, L., Borsook, D., Boas, D. A. (2015). Short separation
regression improves statistical significance and better localizes the hemodynamic response obtained by near-infrared
spectroscopy for tasks with differing autonomic responses. Neurophotonics 2(3), 035005.
Data: OpenNeuro ds005776 v1.0.1 ("Electrical_Thermal_FingerTapping_2015"), doi:10.18112/OPENNEURO.DS005776.V1.0.1.

11 right-handed men: 46 recordings (electrical 11, finger tapping 13, imagery 10, thermal 11, resting 1), 6.91 h.
CW6 (TechEn; 690/830 nm), 50 Hz, 15 sources and 31 detectors: 51 channels (36 at 30 mm, 15 short at 8 mm), all kept.
Missing samples (3 non-positive values in one file) follow the shared rule (fill_or_drop).
Rejected for the main set: no 3D positions. The files store only the 2D AtlasViewer probe design (sourcePos2D and
detectorPos2D, mm; z = 0); coordsystem.json names CapTrak but gives no landmarks. Positions are kept as provided.
"""

from pathlib import Path

import mne

from loaders.utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record


def load_thermal_data(main_dir=DATA_ROOT / "thermal", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*.snirf")):
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw, allow_2d=True)
        record["subject_id"] = path.name.split("_")[0]                      # sub-356 ... sub-3xx
        record["coord_frame"] = "2D AtlasViewer probe design (z = 0), mm; coordsystem.json says CapTrak, no landmarks"
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_thermal_data()
