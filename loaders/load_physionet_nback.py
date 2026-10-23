"""
Csipo, T., Lipecz, A., Mukli, P., Bahadli, D., Abdulhussein, O., Owens, C. D., Tarantini, S., Hand, R. A.,
Yabluchanska, V., Kellawan, J. M., Sorond, F., James, J. A., Csiszar, A., Ungvari, Z. I., Yabluchanskiy, A. (2021).
Increased cognitive workload evokes greater neurovascular coupling responses in healthy young adults. PLoS ONE 16(5),
e0250043. doi:10.1371/journal.pone.0250043.
Data: PhysioNet "Mental workload during n-back task captured by transcranial Doppler (TCD) sonography and functional
near-infrared spectroscopy (fNIRS) monitoring" v1.0 (doi:10.13026/zfb2-1g43, CC BY-NC-SA 4.0); only NIRS_data is used.

14 healthy young adults performed an n-back task (0-back, 1-back, 0-back, 2-back; ~134 s each); fNIRS and transcranial
Doppler were recorded on separate visits. 12 fNIRS recordings are shared (SUBJ003 and SUBJ008 are absent), one per
participant, ~10 min each. The paper excluded two participants from its fNIRS analysis; neither the paper nor PhysioNet
says whether they are the absent ones (the TCD data lack SUBJ007 and SUBJ008).
NIRScoutX (NIRx; LED sources, 760/850 nm), 3.91 Hz, NIRStar 15.0 folders: 16 sources and 16 detectors on a 128-port
EasyCap forming 48 channels (~30 mm) over the prefrontal cortex and the medial motor and somatosensory cortices;
all kept. MNE cannot read the folders (it fails on the event file), so the intensities are read with
utils.read_nirstar.
Positions: each optode is placed at the NIRx ICBM152 position (MNI) of its 10-5 label (PhysioNet description), as for
the other NIRx datasets with labels. Detector 10 is listed as "CC4", which is not a 10-5 label; the probe geometry in
probeInfo.mat (fitted to the other labels, median residual 5 mm) puts it at FC4, the position its four neighbors
(F4, FC6, C4, FC2) imply. Detector 8 (AFF6h) is in no channel.
"""

import json
from pathlib import Path

import numpy as np

from .utils import DATA_ROOT, NIRX_ICBM152, fill_or_drop, print_summary, raw_to_record, read_nirstar

SOURCES = ["F3", "AF7", "AF3", "Fz", "Fpz", "AF4", "F4", "AF8", "FC6", "C4", "FC2", "CP2", "FC1", "CP1", "C3", "FC5"]
DETECTORS = ["F5", "F1", "Fp1", "AFz", "F2", "Fp2", "F6", "AFF6h", "C6", "FC4", "CP4", "C2", "C1", "FC3", "CP3", "C5"]


def optode_xyz():
    """``{optode: MNI xyz in metres}``: the NIRx ICBM152 position of each optode's 10-5 label."""
    table = json.loads(NIRX_ICBM152.read_text())                           # 10-5 label -> MNI (mm)
    optodes = {f"S{i + 1}": table[label] for i, label in enumerate(SOURCES)}
    optodes |= {f"D{i + 1}": table[label] for i, label in enumerate(DETECTORS)}
    return {name: np.array(xyz) / 1000.0 for name, xyz in optodes.items()}


def load_physionet_nback_data(main_dir=DATA_ROOT / "physionet_nback", verbose=True):
    """Return ``{recording_name: record}`` for every NIRStar folder in ``main_dir``."""
    outputs = {}
    xyz = optode_xyz()
    for folder in sorted(Path(main_dir).glob("SUBJ*")):
        raw = read_nirstar(folder)                                          # 48 channels x 2 wavelengths, 3.91 Hz
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw, optode_xyz=xyz, fix_file_positions=True)  # the files have no positions
        record["subject_id"] = folder.name                                  # SUBJ001 ... SUBJ014
        outputs[folder.name] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_physionet_nback_data()
