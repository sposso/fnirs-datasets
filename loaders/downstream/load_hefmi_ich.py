"""
Shi, J., Chen, D., Zhao, X., Zhao, Z., Li, S., Xu, Y., Ding, T., Zhu, Z., Zhang, P., Ye, Q., Tang, Y.,
Zhang, P., Tao, B., Tang, Z. (2025). HEFMI-ICH: a hybrid EEG-fNIRS motor imagery dataset for
brain-computer interface in intracerebral hemorrhage. Scientific Data 12, 1816.
doi:10.1038/s41597-025-06100-7. figshare doi:10.6084/m9.figshare.28955456.v4

37 participants (17 healthy, 20 intracerebral hemorrhage patients) x 1-3 sessions of left/right-hand
motor imagery: 99 recordings of ~15-18 min at 11 Hz.
NirScan: 32 sources and 30 detectors forming 90 channels (30 mm) over the prefrontal, motor and
association cortices; no short channels.
The files carry no participant labels; the conversion groups them by their figshare order, which gives
the paper's 20 patients (sub-01 to sub-20, 49 recordings) and 17 healthy participants (sub-21 to sub-37,
50 recordings), the same split as GeoNIRS (1470 and 1500 trials).
The original Homer .nirs files hold raw intensity at 730/808/850 nm. scripts/convert_hefmi_ich.py
converts them to SNIRF with 730 and 850 nm only (MNE's Beer–Lambert takes two wavelengths) and the
files' flat 2D layout, which encodes the real 30-mm distances used by Beer–Lambert.
After Beer–Lambert, each optode gets the MNI coordinates of the paper's Tables 4 and 5, used as given:
they lie on the brain surface, ~16 mm below the scalp.
Events: the markers of the .nirs files, kept by the conversion as "marker_<column>". Each trial starts with
marker_9 or marker_11 (the two imagery types; which hand is which is not documented), followed ~12 s later by
marker_10 or marker_12 (paper: 2-s cue, then 10 s of imagery). Downstream task (GeoNIRS): healthy vs ICH,
one sample per marker_9 / marker_11 trial, labelled with the participant's group.
"""

from pathlib import Path

import numpy as np

from ..utils import DATA_ROOT, print_summary, snirf_to_record
from . import epochs

# Optode MNI coordinates in mm (paper, Table 4: sources; Table 5: detectors)
OPTODE_MNI = {
    "S1": (71.5, -37.4, 3.7), "S2": (58.9, 13.4, -13.3),
    "S3": (29.0, 59.1, -19.2), "S4": (-23.2, 59.4, -20.5),
    "S5": (-56.8, 15.8, -17.6), "S6": (-71.7, -35.1, 3.8),
    "S7": (67.4, -4.5, 27.7), "S8": (48.8, 51.1, 5.3),
    "S9": (2.9, 70.6, -4.2), "S10": (-48.3, 51.9, 6.3),
    "S11": (-65.6, -0.1, 27.3), "S12": (48.8, 24.5, 43.1),
    "S13": (32.2, 54.8, 29.5), "S14": (-29.1, 57.8, 29.6),
    "S15": (-46.6, 29.2, 43.1), "S16": (57.3, -30.3, 55.6),
    "S17": (-55.1, -23.5, 59.3), "S18": (29.5, -0.8, 69.6),
    "S19": (-28.1, 1.1, 70.5), "S20": (44.4, -88.2, 19.8),
    "S21": (22.7, -85.5, 49.1), "S22": (-4.7, -98.5, 22.7),
    "S23": (-24.3, -83.7, 50.3), "S24": (-47.2, -86.0, 18.5),
    "S25": (22.3, -106.1, -1.2), "S26": (-63.6, -51.1, 40.4),
    "S27": (62.2, -56.7, 39.2), "S28": (-24.5, -105.3, -3.8),
    "S29": (9.8, -1.2, 76.4), "S30": (-19.2, -47.4, 79.6),
    "S31": (19.6, -48.2, 78.9), "S32": (-8.4, -1.8, 76.3),
    "D1": (69.1, -11.0, -4.2), "D2": (50.9, 45.6, -15.5),
    "D3": (2.6, 62.9, -25.0), "D4": (-49.7, 48.3, -16.2),
    "D5": (-68.0, -8.5, -7.8), "D6": (69.5, -31.8, 34.1),
    "D7": (60.6, 24.7, 16.5), "D8": (31.6, 66.1, -2.2),
    "D9": (-28.7, 67.8, -1.9), "D10": (-60.2, 27.6, 15.3),
    "D11": (-69.1, -28.0, 36.6), "D12": (56.8, -1.1, 50.7),
    "D13": (31.9, -29.7, 74.3), "D14": (4.0, 63.3, 31.4),
    "D15": (-31.3, -25.5, 73.8), "D16": (-52.5, 4.8, 51.5),
    "D17": (48.9, -52.2, 57.5), "D18": (-48.7, -47.9, 59.3),
    "D19": (22.5, -100.3, 22.4), "D20": (43.5, -77.9, 44.2),
    "D21": (-3.9, -82.0, 48.0), "D22": (-24.7, -99.5, 21.8),
    "D23": (-47.6, -72.9, 45.6), "D24": (11.3, -30.6, 80.3),
    "D25": (-6.1, -105.3, 1.6), "D26": (65.6, -60.0, 6.5),
    "D27": (-65.8, -58.5, 12.6), "D28": (-9.7, -28.8, 80.3),
    "D29": (15.7, 21.6, 66.9), "D30": (-16.8, 21.3, 67.0),
}
OPTODE_XYZ = {name: np.array(xyz) / 1000.0 for name, xyz in OPTODE_MNI.items()}  # mm -> m
GROUP = {f"sub-{i:02d}": "ICH" if i <= 20 else "healthy" for i in range(1, 38)}


CONDITION = (2.0, 10.0)  # start after the event and duration (s); paper (Fig. 2): 2-s cue at the marker, then 10 s of motor imagery, then 15 s of rest

def load_hefmi_ich_data(main_dir=DATA_ROOT / "hefmi_ich", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/*.snirf")):
        record = snirf_to_record(path, optode_xyz=OPTODE_XYZ)          # the SNIRF markers become the events
        record["subject_id"] = path.parts[-2]  # sub-XX
        record["group"] = GROUP[record["subject_id"]]
        record["event_labels"] = {"marker_9": record["group"], "marker_11": record["group"]}
        record["epochs"], record["labels"] = epochs(record, *CONDITION)
        outputs[path.stem] = record

    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_hefmi_ich_data()
