"""Fix the stroke recovery SNIRF files (Medvedeva et al. 2026, figshare 28904942) so that MNE reads them correctly.

The same three problems are in all 475 files:
1. dataTimeSeries is stored as channels x time (140 x T) instead of time x channels, so MNE reads 140 samples.
2. The data are optical density ("fNIRS data are recorded as optical density values", paper), but are labeled as
   raw intensity (dataType 1, "raw-DC"); they are relabeled as processed optical density (99999, "dOD").
3. Detectors 19-21 are stored at P8, P4 and P7, but the channel list uses them as P4, P3 and Pz: the authors removed
   the malfunctioning detector and its symmetric counterpart (P7, P8; paper) and renumbered the detectors in the
   channel list but not in the positions. With the stored positions 9 channels are 64-137 mm long; with P4, P3 and
   Pz every channel is 30-41 mm long, as in the paper's 3-cm montage (Fig. 2). The positions of P4, P3 and Pz are
   taken from the file's own landmark table (NIRx ICBM152, mm).

Run once from the repository root:  python -m scripts.fix_stroke_recovery
Writes $FNIRS_DATA_ROOT/stroke_recovery/<name>.snirf; the original files are not touched.
"""

import shutil

import h5py
import numpy as np

from loaders.utils import DATA_ROOT

SRC = DATA_ROOT / "stroke_recovery_download/snirf_data/FNIRS_SNIRF"  # files as downloaded
DST = DATA_ROOT / "stroke_recovery"                                 # fixed copies
DETECTOR_LABELS = {19: "P4", 20: "P3", 21: "Pz"}  # detector index in the channel list: its real 10-20 position

DST.mkdir(exist_ok=True)
for path in sorted(SRC.glob("*.snirf")):
    out = DST / path.name
    shutil.copyfile(path, out)                                         # copy, then edit only the copy
    with h5py.File(out, "a") as f:
        data = f["nirs/data1"]
        x = data["dataTimeSeries"][()]
        del data["dataTimeSeries"]
        data["dataTimeSeries"] = x.T                                   # 1. time x channels

        for name in [k for k in data if k.startswith("measurementList")]:
            del data[name]["dataType"], data[name]["dataTypeLabel"]
            data[name]["dataType"] = np.int32(99999)                   # 2. processed data ...
            data[name]["dataTypeLabel"] = np.bytes_("dOD")             #    ... optical density

        probe = f["nirs/probe"]
        labels = [label.decode() for label in probe["landmarkLabels"][()]]
        landmarks = probe["landmarkPos3D"][()][:, :3]
        detectors = probe["detectorPos3D"][()]
        for index, label in DETECTOR_LABELS.items():
            detectors[index - 1] = landmarks[labels.index(label)]      # 3. real detector positions
        probe["detectorPos3D"][...] = detectors
print(f"{len(list(DST.glob('*.snirf')))} files in {DST}")
