"""
Rybář, M., Poli, R., Daly, I. (2025). Simultaneous EEG and fNIRS recordings for semantic decoding of imagined animals
and tools. Scientific Data 12, 613. doi:10.1038/s41597-025-04967-0. OpenNeuro ds004514 (v1.1.2).

12 adults saw images of animals and tools and, after each, silently named the object and imagined seeing it, hearing
it and touching it (3 s each); one ~70-min recording per participant (15 blocks, 180 trials). NIRx NIRScoutXP,
785/830 nm, with two left-hemisphere montages (paper, Table 1): frontal (sub-01 to sub-06; 8 sources, 4 detectors,
14 channels, 7.81 Hz) and temporal-parietal (sub-07 to sub-12; 7 sources, 4 detectors, 11 channels, 8.93 Hz).
The files store the NIRx template positions in a CapTrak frame. Each optode is identified with its 10-5 position in
the paper's Fig. 2 (the only labelling consistent with the channel list; after a rotation and a shift the file
positions lie within 0.2 mm of the template positions of these labels) and gets its NIRx ICBM152 MNI position.
Events: the BIDS events.tsv (the SNIRF stim only holds trigger codes): each trial is an image (animal_image or
tool_image), a mask, then four ~3-s tasks (tactile, visual and auditory imagery, silent naming) from ~1.8 to 14.3 s;
images are ~18 s apart. Downstream task (GeoNIRS): animal vs tool, one sample per image.
"""

from pathlib import Path

import mne
import numpy as np

from ..utils import DATA_ROOT, print_summary, raw_to_record
from . import bids_events, epochs

CONDITION = (0.0, 14.3)  # start after the event and duration (s); paper: image 0.6 s, mask 0.6 s, blank 0.5 s, four 3-s tasks 0.2 s apart

FRONTAL_SUBJECTS = {f"sub-{i:02d}" for i in range(1, 7)}  # the others used the temporal montage

# optode: (10-5 position in the paper's Fig. 2, NIRx ICBM152 MNI coordinates in mm)
FRONTAL = {
    "S1": ("AFF5h", (-51.040, 65.913, 16.492)), "S2": ("FFC2h", (16.566, 46.584, 75.853)),
    "S3": ("AFF1h", (-12.859, 74.475, 48.143)), "S4": ("FFC3h", (-46.943, 42.977, 58.568)),
    "S5": ("FCC1h", (-19.904, 10.258, 92.246)), "S6": ("AFp2", (13.465, 87.104, 10.414)),
    "S7": ("FFT7h", (-74.635, 31.719, -1.932)), "S8": ("FCC5h", (-76.788, 4.642, 41.212)),
    "D1": ("FFC1h", (-17.372, 46.329, 75.763)), "D2": ("AFp1", (-13.391, 87.129, 10.402)),
    "D3": ("FFC5h", (-66.426, 37.778, 30.883)), "D4": ("AFF2h", (12.485, 74.553, 48.159)),
}
TEMPORAL = {
    "S1": ("FFT9h", (-77.974, 27.617, -39.812)), "S2": ("FTT7h", (-82.630, 0.747, 2.398)),
    "S3": ("CCP5h", (-81.397, -30.295, 47.238)), "S4": ("TPP7h", (-78.902, -62.370, 8.661)),
    "S5": ("PPO9h", (-65.352, -86.533, -24.847)), "S6": ("CPP3h", (-52.376, -66.331, 76.844)),
    "S7": ("PPO5h", (-56.629, -93.250, 35.816)),
    "D1": ("TTP7h", (-86.306, -31.193, 5.631)), "D2": ("TPP9h", (-77.899, -60.455, -30.540)),
    "D3": ("FTT9h", (-81.992, -2.539, -37.589)), "D4": ("CPP5h", (-73.325, -64.597, 45.833)),
}
FRONTAL_XYZ = {name: np.array(xyz) / 1000.0 for name, (_, xyz) in FRONTAL.items()}    # mm -> m
TEMPORAL_XYZ = {name: np.array(xyz) / 1000.0 for name, (_, xyz) in TEMPORAL.items()}


def load_tools_animals_data(main_dir=DATA_ROOT / "tools_animals", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*.snirf")):
        subject = path.parts[-3]                                       # sub-XX
        optode_xyz = FRONTAL_XYZ if subject in FRONTAL_SUBJECTS else TEMPORAL_XYZ
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        raw.set_annotations(bids_events(path))                              # names instead of the trigger codes
        record = raw_to_record(raw, optode_xyz=optode_xyz)
        record["subject_id"] = subject
        record["event_labels"] = {"animal_image": "animal", "tool_image": "tool"}
        record["epochs"], record["labels"] = epochs(record, *CONDITION)
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_tools_animals_data()
