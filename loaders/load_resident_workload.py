"""
Torres-Cuevas, G. E., Martínez-Jiménez, M. A., Guevara, E. (2025). Resident physician mental workload assessed with
fNIRS. Zenodo. doi:10.5281/zenodo.15186569 (all versions; this version 10.5281/zenodo.15186570; files restricted; manual download, without the videos).
No paper is published; the dataset ships an unfinished draft data descriptor (derivatives/protocol.pdf).

30 resident physicians recorded at the start (0 h), middle (12 h) and end (24 h) of a 24-h shift, each time during
a 5-min laparoscopic training task (origami box folding in a box trainer; lap01-03) and a 6-min eyes-closed rest
(resting01-03): 180 recordings. Artinis Brite MKII, CW, 757/843 nm, 25 Hz (21 files at 75 Hz), 10 sources and
8 detectors over the prefrontal cortex: 22 channels at 3 cm and 2 short channels.
Positions: the files hold only a flat 2D layout (in cm although labeled mm); the authors give one MNI point per
channel (derivatives/optodes_MNI_coordinates.csv), used as given and matched by channel number (the file's channel
order: two rows of the table carry wrong source/detector labels). Beer-Lambert uses the 3-cm separation of the paper.
Dropped: the 2 short channels (S3_D1, S10_D8; no MNI point) and channel 5 (S2_D2), whose MNI point is a copy of
channel 10's (S2_D4) although the two channels are 16 mm apart on the cap.
"""

import csv
from pathlib import Path

import mne
import numpy as np

from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

SD_DISTANCE = 0.030                         # m, separation of the long channels (paper)
DROPPED = {"S3_D1", "S10_D8", "S2_D2"}      # short channels without MNI point; channel 5 with a copied MNI point


def channel_mni(raw, table):
    """``{"S1_D1": MNI xyz in metres}`` from the authors' table, row i = i-th channel of the file."""
    rows = list(csv.reader(open(table, encoding="utf-8-sig")))[2:]         # two header lines
    pairs = list(dict.fromkeys(name.split(" ")[0] for name in raw.ch_names))  # channels in file order
    return {pair: np.array(row[3:6], float) / 1000.0 for pair, row in zip(pairs, rows) if row[3]}


def load_resident_workload_data(main_dir=DATA_ROOT / "resident_workload", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("*/bids_compatible/sub-*/nirs/*_nirs.snirf")):
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        channel_pos = channel_mni(raw, path.parents[2] / "derivatives" / "optodes_MNI_coordinates.csv")
        raw.drop_channels([name for name in raw.ch_names if name.split(" ")[0] in DROPPED])
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw, channel_pos=channel_pos, sd_distance=SD_DISTANCE)
        record["subject_id"] = path.name.split("_")[0]                      # e.g. sub-01, the same person in all
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_resident_workload_data()
