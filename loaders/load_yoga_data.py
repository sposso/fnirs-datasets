"""
Goodrick, M., Limón Cano, D. E., Guevara, E. (2026). Multimodal Dataset of a 30-Day Yoga Intervention: fNIRS,
Biochemical, Postural, and Subjective Outcomes. Zenodo, doi:10.5281/zenodo.20560768 (v1.0.0; no paper found).

Adults in a 30-day yoga intervention group (n = 26) or a control group, recorded before and after the 30 days in three
tasks: 6 min of rest, 9 min of affective image viewing (IAPS) and 4.5 min of breath holding (protocol.pdf). The
shared zips hold 232 recordings with a participant ID (the zip descriptions announce more for the pre-control
sessions than the zips contain); the 5 recordings stored as "sub-unknown" are skipped, since their participant cannot
be identified. Device not reported: 10 sources and 8 detectors (757/843 nm), 24 channels over the prefrontal cortex,
25 or 75 Hz; raw intensity.
Positions: the SNIRF files hold only a flat 2D layout, so each channel is placed at the MNI coordinates given by the
authors (optodes_MNI_coordinates.csv), used as given. The two short-separation channels (S3-D1, S10-D8) have no
coordinates and are dropped, as for the resident-workload dataset. In the 2D layout the long channels are 3.0 units
long and the short ones 1.0, i.e. the layout is in cm although labeled mm: Beer–Lambert uses 30 mm for every channel.
"""

import csv
from pathlib import Path

import mne
import numpy as np

from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

SD_DISTANCE = 0.030  # m


def channel_mni(path):
    """``{"S1_D1": MNI xyz in metres}`` from the authors' CSV (channels without coordinates are left out)."""
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))[2:]           # two header lines
    return {f"S{r[1]}_D{r[2]}": np.array([float(v) for v in r[3:6]]) / 1000.0 for r in rows if r[3].strip()}


def load_yoga_data(main_dir=DATA_ROOT / "yoga", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file with a participant ID in ``main_dir``."""
    positions = channel_mni(Path(main_dir) / "optodes_MNI_coordinates.csv")
    outputs = {}
    for path in sorted(Path(main_dir).glob("*/*/bids_compatible/sub-*/nirs/*.snirf")):
        subject = path.parts[-3]                                            # e.g. sub-15
        if subject == "sub-unknown":
            continue
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        raw.pick([name for name in raw.ch_names if name.split(" ")[0] in positions])  # drop the 2 short channels
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        if raw is None:
            continue
        record = raw_to_record(raw, channel_pos=positions, sd_distance=SD_DISTANCE)
        record["subject_id"] = subject
        outputs[f"{path.parts[-6]}_{subject}"] = record                      # e.g. Pre_Ctrl_Images_sub-15
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_yoga_data()
