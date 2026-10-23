"""
Cockx, H., Oostenveld, R., Tabor, M., Savenco, E., van Setten, A., Cameron, I., van Wezel, R. (2023). fNIRS is
sensitive to leg activity in the primary motor cortex after systemic artifact correction. NeuroImage 269, 119880.
doi:10.1016/j.neuroimage.2023.119880.
Data: Donders Repository DSC_62003212_03_066 v1, "Automatic and non-automatic finger and leg movements measured with
functional near-infrared spectroscopy (fNIRS)" (doi:10.34973/vesb-mh30, CC BY-SA 4.0), BIDS folder bids_raw.

24 adults, 8 runs of 7.5-14 min each: right-hand finger tapping or right-foot stepping of a 12-digit sequence, either
practiced for 7 days (automatic) or learned on the day (non-automatic), each with and without a letter-counting dual
task. Two Artinis Brite24 devices (LED sources, 760/850 nm, photodiode detectors), 50 Hz: 24 long (~30 mm) and
12 short (~10 mm) channels over left M1/PMC, bilateral dlPFC and bilateral PPC; all are kept.
Excluded: participants flagged by the authors in participants.tsv (included = 0: sub-10, more than half of the channels
with SCI < 0.7). Dropped: sub-69's short channel S21_D9, whose source S21 is stored at the position of detector D9
(0-mm separation).
Data: optical density as exported from Artinis through FieldTrip, which uses base-10 logarithms (FieldTrip's Artinis
toolbox: cfg.logarithm default 'base10'); multiplied by ln(10) because MNE's Beer–Lambert expects natural-log OD.
Positions: digitized for each participant from a 3D scan of the head, in mm (BIDS "CTF" frame), with the nasion,
inion, preauricular points and Cz (coordsystem.json); fitted to fsaverage and returned in MNI by utils.optode_mni.
"""

import csv
from pathlib import Path

import mne
import numpy as np

from .utils import DATA_ROOT, optode_mni, print_summary, raw_to_record

DROP_PAIRS = {"sub-69": {"S21_D9"}}  # source S21 stored at detector D9's position (0-mm separation)


def load_finger_leg_data(main_dir=DATA_ROOT / "finger_leg", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    with open(Path(main_dir) / "participants.tsv") as f:
        excluded = {row["participant_id"] for row in csv.DictReader(f, delimiter="\t") if row["included"] == "0"}
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*_nirs.snirf")):
        subject = path.name.split("_")[0]                                   # e.g. sub-06
        if subject in excluded:
            continue
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        optode_xyz = optode_mni(raw, path.parent / f"{subject}_coordsystem.json")
        raw.apply_function(lambda x: x * np.log(10), channel_wise=False)    # base-10 OD -> natural-log OD
        record = raw_to_record(raw, drop_pairs=DROP_PAIRS.get(subject, ()), optode_xyz=optode_xyz)
        record["subject_id"] = subject
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_finger_leg_data()
