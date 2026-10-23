"""
Khan, H., Nazeer, H., Mirtaheri, P. (2026). Open access individual finger movement dataset with fNIRS. Frontiers in
Human Neuroscience 20, 1747655. doi:10.3389/fnhum.2026.1747655. Data: figshare private link from the paper
(FingerTappingDataset_Published2025.zip, manual download).

25 right-handed adults tapped the fingers of the right hand one at a time (10-s blocks, thumb to little finger,
3 trials): 69 runs of ~6 min (6.81 h). NIRScout (760/850 nm, 3.91 Hz; S25: NIRSport2, 10.17 Hz), 16 sources and
16 detectors forming 48 channels (~30 mm) over both motor cortices; all kept.
The SNIRF files (Satori export, dataType 99999) hold raw intensity (label RAW, both wavelengths) followed by dOD, HbO
and HbR; only the raw intensity is used (MNE cannot read the files). The authors removed masked channels and trimmed
the recordings to the stimuli ("Experimental notes.txt"); the remaining 48 channels are identical in every run.
Positions: each optode is placed at the NIRx ICBM152 position (MNI) of its 10-10 label (paper, Table 3), as in
Shin 2018; the 3D positions in the files (no landmarks, unknown frame) agree with these labels within ~5 mm.
"""

import json
from pathlib import Path

import h5py
import mne
import numpy as np

from .utils import DATA_ROOT, NIRX_ICBM152, print_summary, raw_to_record, set_optode_positions

SOURCES = ["F1", "C1", "FC3", "CP3", "F5", "C5", "FT7", "TP7", "FT8", "TP8", "F6", "C6", "FC4", "CP4", "F2", "C2"]
DETECTORS = ["FC1", "CP1", "F3", "C3", "FC5", "CP5", "F7", "T7", "F8", "T8", "FC6", "CP6", "F4", "C4", "FC2", "CP2"]


def read_raw_intensity(path):
    """Raw intensity (the RAW measurements) of one Satori SNIRF file as an MNE Raw, optodes at their MNI positions."""
    with h5py.File(path, "r") as f:
        data = f["nirs/data1/dataTimeSeries"][()]
        time = f["nirs/data1/time"][()]
        wavelengths = f["nirs/probe/wavelengths"][()]
        names, columns = [], []
        for i in range(data.shape[1]):
            m = f[f"nirs/data1/measurementList{i + 1}"]
            if np.ravel(m["dataTypeLabel"][()])[0] in (b"RAW", "RAW"):
                s, d, w = (int(np.ravel(m[k][()])[0]) for k in ("sourceIndex", "detectorIndex", "wavelengthIndex"))
                names.append(f"S{s}_D{d} {wavelengths[w - 1]:.0f}")
                columns.append(i)
    order = sorted(range(len(names)), key=lambda k: (names[k].split(" ")[0], names[k]))  # wavelengths of a pair together
    info = mne.create_info([names[k] for k in order], 1 / np.median(np.diff(time)), "fnirs_cw_amplitude")
    for ch in info["chs"]:
        ch["loc"][9] = float(ch["ch_name"].split(" ")[1])                  # wavelength (nm)
    raw = mne.io.RawArray(data[:, [columns[k] for k in order]].T, info, verbose=False)
    table = json.loads(NIRX_ICBM152.read_text())                           # 10-5 label -> MNI (mm)
    optodes = {f"S{i + 1}": table[label] for i, label in enumerate(SOURCES)}
    optodes |= {f"D{i + 1}": table[label] for i, label in enumerate(DETECTORS)}
    set_optode_positions(raw.info, {k: np.array(v) / 1000.0 for k, v in optodes.items()})
    return raw


def load_finger_movements_data(main_dir=DATA_ROOT / "finger_movements", verbose=True):
    """Return ``{recording_name: record}`` for every raw SNIRF file (``*_TRIM.snirf``) in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("S*/S*_TRIM.snirf")):
        record = raw_to_record(read_raw_intensity(path))
        record["subject_id"] = path.parent.name                             # S01 ... S25
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_finger_movements_data()
