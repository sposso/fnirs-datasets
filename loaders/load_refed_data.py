"""
Ning, X., Wang, J., Feng, Z., Xin, T., Zhang, S., Zhang, S., Lian, Z., Ding, Y., Lin, Y., Jia, Z. (2025).
REFED: A Subject Real-time Dynamic Labeled EEG-fNIRS Synchronized Recorded Emotion Dataset. NeurIPS 2025
Datasets and Benchmarks Track. Data: Hugging Face REFED2025/REFED-dataset (CC BY-NC-SA).

32 participants watching 15 emotional film clips (1-3 min each) while EEG and fNIRS were recorded; each
clip is one recording (480 recordings). Shimadzu LABNIRS (780/805/830 nm) at 47.62 Hz: 18 sources and
18 detectors forming 51 channels (30 mm) over the prefrontal cortex and both temporal lobes.
We use the HbO/HbR computed by the Shimadzu software, as concentration x path length (mM*cm, as in the
other Shimadzu dataset), divided by 18 cm (30 mm x partial pathlength factor 6) to get M.
Channel positions are the paper's Table 3 (fNIRS_coordinates.csv), in a head frame with x towards the
left ear and y towards the back of the head: we flip x and y and convert them to MNI with head_to_mni.
The channels the authors marked as bad for each participant (fNIRS_reservations.csv: 0) are dropped.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from .utils import DATA_ROOT, PPF, hb_to_record, head_to_mni, print_summary

SFREQ = 47.62        # Hz
SD_DISTANCE = 0.030  # m
TO_MOLAR = 1e-3 / (SD_DISTANCE * 100 * PPF)  # mM*cm -> M, dividing by 18 cm


def read_positions(path):
    """Channel MNI positions (m) from the paper's Table 3: flip x (left) and y (back), then head -> MNI."""
    table = pd.read_csv(path)                                  # Channel, Source, Detector, X, Y, Z (mm)
    head = np.c_[-table.X, -table.Y, table.Z] / 1000.0         # x towards the right ear, y towards the nose
    return head_to_mni(head)


def read_good_channels(path):
    """Per participant, the indices of the channels the authors kept (fNIRS_reservations.csv: 1 kept, 0 bad)."""
    table = pd.read_csv(path, index_col="sub_id")              # columns ch-1 ... ch-51
    return {sub: np.flatnonzero(row.to_numpy() == 1) for sub, row in table.iterrows()}


def load_refed_data(main_dir=DATA_ROOT / "refed", verbose=True):
    """Return ``{recording_name: record}`` for every clip of every participant in ``main_dir``."""
    channel_xyz = read_positions(Path(main_dir) / "fNIRS_coordinates.csv")
    good_channels = read_good_channels(Path(main_dir) / "fNIRS_reservations.csv")
    outputs = {}
    for sub_dir in sorted(Path(main_dir, "data").iterdir(), key=lambda p: int(p.name)):
        subject = f"sub-{int(sub_dir.name):02d}"
        good = good_channels[int(sub_dir.name)]
        clips = sio.loadmat(sub_dir / "fNIRS_videos.mat")      # video_1..video_15
        for k in range(1, 16):
            signals = clips[f"video_{k}"][:, good]             # (HbO, HbR, HbT, Abs780, Abs805, Abs830) x channels x time
            record = hb_to_record(signals[0] * TO_MOLAR, signals[1] * TO_MOLAR, SFREQ, channel_xyz[good],
                                  np.full(len(good), SD_DISTANCE))
            record["subject_id"] = subject
            outputs[f"{subject}_video-{k:02d}"] = record

    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_refed_data()
