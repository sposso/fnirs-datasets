"""
Yücel, M. A., Selb, J., Boas, D. A., Cash, S. S., Cooper, R. J. (2014). Reducing motion artifacts for long-term
clinical NIRS monitoring using collodion-fixed prism-based optical fibers. NeuroImage 85, 192-201.
Data: OpenNeuro ds005929 v1.0.1 ("Motion-Yucel2014"), doi:10.18112/OPENNEURO.DS005929.V1.0.1.

7 adults (sub-6016 ... sub-6023, no sub-6017; the paper reports 5) performing deliberate head and face movements
(reading aloud, nodding, twisting, head shaking, eyebrow raising): one recording each of 6-13 min (three contain the
task twice). CW6 (TechEn; 690/830 nm), 50 Hz, 6 sources and 10 detectors: 14 channels, all kept.
Rejected for the main set: no 3D positions. The files store a flat 2D probe drawing (z = 0) relative to the device
(BIDS "Other"), without landmarks; positions are kept as provided, in the declared unit (mm). The drawing is in cm
(every source-detector distance reads 3.0 mm), so Beer-Lambert uses the paper's 30 mm (sd_distance).
The files hold an empty landmark array without labels, which MNE cannot read: each file is read from a temporary copy
without the empty landmark fields (data unchanged).
"""

import shutil
import tempfile
from pathlib import Path

import h5py
import mne

from loaders.utils import DATA_ROOT, print_summary, raw_to_record


def read_snirf(path):
    """Raw intensity of one file, read from a temporary copy without its empty landmark fields."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / path.name
        shutil.copyfile(path, copy)
        with h5py.File(copy, "r+") as f:
            del f["nirs/probe/landmarkPos2D"], f["nirs/probe/landmarkPos3D"]
        return mne.io.read_raw_snirf(copy, verbose="error").load_data(verbose="error")


def load_motion_data(main_dir=DATA_ROOT / "motion", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-1/nirs/*.snirf")):
        record = raw_to_record(read_snirf(path), sd_distance=0.03, allow_2d=True)
        record["subject_id"] = path.name.split("_")[0]                      # sub-6016 ... sub-6023
        record["coord_frame"] = "2D probe drawing relative to the device (z = 0), declared mm, no landmarks"
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_motion_data()
