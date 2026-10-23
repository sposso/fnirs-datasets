"""
Wang, C., Zhang, Q., Zhong, Y., Zhang, X. (2025). Differential neural correlates of two cognitive reappraisal strategies:
evidence from functional near-infrared spectroscopy. SSRN preprint 5163908, doi:10.2139/ssrn.5163908.
Data: Wang, C. (2023), Science Data Bank doi:10.57760/sciencedb.13806 (V1, CC BY 4.0).

25 undergraduates regulating responses to IAPS pictures (reactivity, detached, positive reappraisal; 6 x 20 trials) (October-November 2019): 26 NIRStar recordings of
4.6-45.4 min (40-45 min except one) from 25 participants. One participant has two sessions on 2019-11-04: a first one stopped after 4.6 min
and a full one; both are kept under the same subject_id.
NIRScoutX (NIRx; laser sources, APD detectors, 785/830 nm), 3.91 Hz, 16 sources and 12 detectors on standard 10-20
positions over the prefrontal cortex and anterior temporal lobes: 41 channels (~30 mm); all are kept.
The paper excluded one female participant (fNIRS fluctuations > 3 SD), not identified in the data: all are kept.
The paper's Table 1 MNI (11 channels, NIRS_SPM cortical projections) lie 15-25 mm deeper than our scalp midpoints;
the table point is nearest the same-numbered channel for 9 of 11 (Ch8, Ch26 ~21-25 mm off tangentially).
Optode positions: probeInfo.mat holds no optode labels, and its coordinates are in the frame of the head model stored
in the same file (geom.NIRxHead), not MNI. Every optode lies exactly (0 mm) on one of the 10-20 points of that head
model (geom.NIRxHead.ext1020sys), which names it; each optode gets the NIRx ICBM152 MNI position of its label (as for
Shin 2018). The montage is identical for every recording.
Folder names hold participants' names: subject_id is an anonymous number (sub-01 ... in folder order).
"""

import json
import re
from pathlib import Path

import mne
import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree

from .utils import DATA_ROOT, NIRX_ICBM152, print_summary, raw_to_record


def optode_labels(probe_path):
    """``{optode: 10-20 label}`` from the head-model 10-20 point each optode of probeInfo.mat sits on."""
    probe = sio.loadmat(probe_path, squeeze_me=True, struct_as_record=False)["probeInfo"]
    system = probe.geom.NIRxHead.ext1020sys
    names = [f"S{i + 1}" for i in range(len(probe.probes.coords_s3))]
    names += [f"D{i + 1}" for i in range(len(probe.probes.coords_d3))]
    distance, index = cKDTree(system.coords3d).query(np.r_[probe.probes.coords_s3, probe.probes.coords_d3])
    assert distance.max() < 1e-6, f"{probe_path}: optode off the 10-20 points"
    return dict(zip(names, system.labels[index]))


def load_emotion_reappraisal_data(main_dir=DATA_ROOT / "emotion_reappraisal", verbose=True):
    """Return ``{recording_name: record}`` for every NIRStar folder in ``main_dir``."""
    template = json.loads(NIRX_ICBM152.read_text())                    # 10-5 label -> MNI xyz (mm)
    subjects, outputs = {}, {}
    for folder in sorted(p for p in Path(main_dir).glob("DATA/*") if p.is_dir()):
        name = re.sub(r"^[\d-]+_\d+", "", folder.name)                  # strip "2019-11-04_001": the participant
        subject_id = subjects.setdefault(name, f"sub-{len(subjects) + 1:02d}")
        labels = optode_labels(next(folder.glob("*_probeInfo.mat")))
        optode_xyz = {optode: np.array(template[label]) / 1000.0 for optode, label in labels.items()}
        raw = mne.io.read_raw_nirx(folder, verbose="error").load_data(verbose="error")
        record = raw_to_record(raw, optode_xyz=optode_xyz)
        record["subject_id"] = subject_id
        outputs[f"{subject_id}_{next(folder.glob('*.wl1')).stem}"] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_emotion_reappraisal_data()
