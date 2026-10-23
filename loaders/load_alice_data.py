"""
Yu, C.-L., Eggleston, R., Zhang, K., Nickerson, N., Sun, X., Marks, R. A., Hu, X., Brennan, J. R., Wellman, H. M.,
Kovelman, I. (2025). Neural processing of children's theory of mind in a naturalistic story-listening paradigm.
Social Cognitive and Affective Neuroscience 20(1), nsaf022. doi:10.1093/scan/nsaf022.
Data: Deep Blue Data, doi:10.7302/zw4w-hp67 (the NIRSfiles/ folder of the download is the dataset folder).

51 children (5.6-11.97 years) listened to the first chapter of Alice's Adventures in Wonderland (12.4 min); one
recording each (12.3-13.0 min), kept whole. TechEn CW6, 690/830 nm, 50 Hz.
Channels: the lab's 46-channel cap, see kovelman_cap.py (the files hold 54 pairs; the 8 without an MNI point are
dropped, as are pairs switched off in the file).
Positions: the paper's own channel MNI estimates (supplementary Table S1; separate child sample, 3D digitizer and MRI),
channels 1-23 on the left and their right-hemisphere homologues (x mirrored) as channels 24-46, in the files'
measurement order. They lie a median 6.3 mm (max 10.4 mm) from the CHMNI points used for the other two datasets.
"""

import numpy as np

from .kovelman_cap import PAIRS, nirs_files, read_raw
from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

TABLE_S1 = [  # left-hemisphere channels 1-23, MNI mm (paper, supplementary Table S1)
    (-56, 50, -17), (-53, 49, -1), (-62, 29, -14), (-59, 33, 5), (-65, 12, -11), (-62, 17, 8), (-68, -4, -9),
    (-64, -1, 11), (-67, -19, -6), (-64, -16, 14), (-63, -33, -3), (-60, -30, 16), (-56, -46, -2), (-53, -43, 17),
    (-46, -59, 1), (-45, -53, 15), (-67, -22, -26), (-63, -36, -23), (-58, -40, -40), (-55, -50, -21), (-45, -63, -18),
    (-51, -54, -38), (-44, -64, -34),
]
POINTS = np.array(TABLE_S1 + [(-x, y, z) for x, y, z in TABLE_S1], float) / 1000.0  # + right homologues; mm -> m
CHANNEL_MNI = {f"S{s}_D{d}": xyz for (s, d), xyz in zip(PAIRS, POINTS)}


def load_alice_data(main_dir=DATA_ROOT / "alice_story", verbose=True):
    """Return ``{recording_name: record}`` for every .nirs file in ``main_dir``."""
    outputs = {}
    for path in nirs_files(main_dir):
        raw = read_raw(path, mni=CHANNEL_MNI)
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw)
        record["subject_id"] = path.stem.split("_")[0]                   # e.g. 1094
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_alice_data()
