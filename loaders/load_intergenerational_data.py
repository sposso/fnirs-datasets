"""
Moffat, R., Cross, E. S. (2026). A longitudinal fNIRS hyperscanning dataset of same generation and intergenerational
relationship development and collaboration. bioRxiv. doi:10.64898/2026.06.26.734715.
Data: OpenNeuro ds008192 v1.0.5 (InterGenSynchrony Dataset); only the fNIRS part and the coordinate tables were
downloaded.

61 dyads (31 intergenerational: an older adult, 69+, with a younger adult; 30 of two younger adults), 6 weekly sessions
of drawing (5 min alone, twice 5 min together, no talking) and a short collaborative task; each person in one recording
of ~20 min (6 sessions were recorded in two parts, kept as separate recordings). Cortivision Photon Cap (760/850 nm,
~5 Hz, upsampled to 7.5 Hz like the other datasets): 12 sources and 10 detectors at 10-5 positions over the left and
right IFG and TPJ (22 channels), plus 4 short channels (sources S13-S16).
Channels flagged "bad" by the authors in channels.tsv (visual inspection: no cardiac oscillation or major motion) are
dropped; every recording is kept with the channels it has left (at least 4 of 26).
Positions: the session's digitized MNI channel positions (sourcedata/hyperscanning/coordinates, from 3D head scans)
where available (447 of 719 sessions, none for the older adults); otherwise the template 10-5 positions stored in the
SNIRF files, identical for every recording. Beer–Lambert uses the template source-detector distances. The BIDS
optodes.tsv files are not used: their positions do not match the montage.
Excluded: the first session of sub-302 and sub-303, recorded with an earlier montage that the files do not describe.
sub-243 ses-05 is kept although the paper lists it as missing (alcohol before the session).
"""

import csv
from pathlib import Path

import mne
import numpy as np

from .utils import DATA_ROOT, print_summary, raw_to_record

EXCLUDED = {"sub-302_ses-01", "sub-303_ses-01"}  # earlier TPJ montage, positions unknown (paper Table 3)


def digitized_positions(main_dir, subject, session):
    """``{"S1_D2": xyz in metres}`` digitized in this session (MNI), or None if it was not digitized."""
    files = list(Path(main_dir, "sourcedata/hyperscanning/coordinates", subject, session).glob("*_optodeCoordinates.csv"))
    if not files:
        return None
    with open(files[0]) as f:
        return {r["channel"].strip(): np.array([float(r["x"]), float(r["y"]), float(r["z"])]) / 1000.0  # mm -> m
                for r in csv.DictReader(f)}


def load_intergenerational_data(main_dir=DATA_ROOT / "intergenerational", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*_nirs.snirf")):
        subject, session = path.parts[-4], path.parts[-3]
        if f"{subject}_{session}" in EXCLUDED:
            continue
        with open(path.with_name(path.name.replace("_nirs.snirf", "_channels.tsv"))) as f:
            status = list(csv.DictReader(f, delimiter="\t"))
        pairs = {r["name"].split(" ")[0] for r in status}                   # "S1_D2 760" -> "S1_D2"
        bad = {r["name"].split(" ")[0] for r in status if r["status"] == "bad"}
        if not pairs - bad:                                                  # every channel flagged
            continue
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        record = raw_to_record(raw, drop_pairs=bad, channel_pos=digitized_positions(main_dir, subject, session))
        record["subject_id"] = subject
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_intergenerational_data()
