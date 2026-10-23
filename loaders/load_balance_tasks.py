"""
Almulla, L., Al-Naib, I., Ateeq, I.S., Althobaiti, M. (2022). Observation and motor imagery balance tasks evaluation:
An fNIRS feasibility study. PLoS ONE 17(3), e0265898. doi:10.1371/journal.pone.0265898.
Data: figshare doi:10.6084/m9.figshare.19131281 (v1; MD5 checked): NIRStar files per recording, named GroupG_SubjectN.

13 healthy adults (4 of the 17 recorded were excluded by the authors for poor data quality and are not shared) watched
and/or imagined static and dynamic standing balance tasks in three experiments: action observation (Group1), action
observation with motor imagery (Group2) and motor imagery (Group3); five 10-s trials of each task with 10-s rests,
one recording of ~4 min per experiment: 39 recordings. NIRSport2 tandem (NIRx; 760/850 nm, 3.47 Hz), 16 sources and
15 detectors (detector 8 unused) over the prefrontal and both motor cortices, 40 channels of 29-31 mm (paper, Fig. 1).
Raw intensity from the .wl1/.wl2 files (utils.read_nirstar); samples flagged as saturated by NIRStar are NaN there
(the unflagged values are in the .nosatflags files) and are handled by utils.fill_or_drop (removes S8_D7 of Group3_Subject4, saturated in 644 of 933 samples).
No other flags.
Positions: probeInfo.mat (NIRx) places every optode exactly (0.00 mm) on a 10-20 position of its ext1020sys table,
the same montage in all 39 files (optodes chosen with fOLD, paper); the labels are mapped to the NIRx ICBM152 table.
Beer–Lambert uses the template distances between the labels (30-46 mm; paper: 29-31 mm), as for the other template
positions.
"""

import json
from pathlib import Path

import numpy as np

from .utils import DATA_ROOT, NIRX_ICBM152, fill_or_drop, print_summary, raw_to_record, read_nirstar, set_optode_positions

SOURCES = ["F3", "AF7", "AF3", "Fz", "Fpz", "AF4", "F4", "AF8", "FC6", "C4", "FC2", "CP2", "FC1", "CP1", "C3", "FC5"]
DETECTORS = ["F5", "F1", "Fp1", "AFz", "F2", "Fp2", "F6", "F8", "C6", "FC4", "CP4", "C2", "C1", "FC3", "CP3", "C5"]


def load_balance_tasks_data(main_dir=DATA_ROOT / "balance_tasks", verbose=True):
    """Return ``{recording_name: record}`` for every recording (``GroupG_SubjectN.hdr``) in ``main_dir``."""
    table = json.loads(NIRX_ICBM152.read_text())                            # 10-5 label -> MNI (mm)
    optodes = {f"S{i + 1}": table[label] for i, label in enumerate(SOURCES)}
    optodes |= {f"D{i + 1}": table[label] for i, label in enumerate(DETECTORS)}
    optodes = {k: np.array(v) / 1000.0 for k, v in optodes.items()}
    outputs = {}
    for hdr in sorted(Path(main_dir).glob("Group*_Subject*.hdr")):
        raw = read_nirstar(hdr)
        set_optode_positions(raw.info, optodes)
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        if raw is None:
            continue
        record = raw_to_record(raw)
        record["subject_id"] = hdr.stem.split("_")[1]                        # Subject1 ... Subject13
        outputs[hdr.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_balance_tasks_data()
