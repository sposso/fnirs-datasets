"""
Satish, G., Taylor, L. M., Arnold, M. P., Gallagher-Shale, J., Krishnamurthy, K., Basura, G. J. (2026). Subjective
tinnitus severity correlates with objective changes in human auditory cortex as measured by functional near-infrared
spectroscopy. OpenNeuro ds007990 v1.0.0, doi:10.18112/openneuro.ds007990.v1.0.0 (no paper found).

63 adults (34 controls, sub-0xx; 29 with tinnitus, sub-1xx), one recording each of 16-20 min (3 min of silence, then
12-s broadband-noise blocks). NIRSport2 (760/850 nm), 24 sources and 31 detectors, 83 channels: 8 short (8-9 mm) and
75 of 28-44 mm, all kept (the 35-mm cutoff is only applied to HD-DOT arrays). 5.09 Hz.
Optode positions: the montage on the NIRx ICBM152 head (the files' 300 10-5 landmarks equal the NIRx ICBM152 table),
i.e. MNI coordinates; kept as stored. No author flags (every channel "good" in channels.tsv).
Events: the BIDS events.tsv (silence, then BBN: 12-s broadband-noise blocks). Groups: participants.tsv
(control, or "experimental" = tinnitus). Downstream task (GeoNIRS): control vs tinnitus, one sample per noise block,
labelled with the participant's group.
"""

import csv
from pathlib import Path

import mne

from ..utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record
from . import bids_events, epochs


CONDITION = (0.0, 12.0)  # start after the event and duration (s); no paper; the dataset events give 12-s broadband-noise blocks

def load_tinnitus_data(main_dir=DATA_ROOT / "tinnitus", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    groups = {row["participant_id"]: {"control": "control", "experimental": "tinnitus"}[row["group"]]
              for row in csv.DictReader(open(Path(main_dir) / "participants.tsv"), delimiter="\t")}
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*.snirf")):
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        raw.set_annotations(bids_events(path))                              # names instead of the stim codes
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw)
        record["subject_id"] = path.name.split("_")[0]                      # sub-001 ... sub-126
        record["group"] = groups[record["subject_id"]]
        record["event_labels"] = {"BBN": record["group"]}
        record["epochs"], record["labels"] = epochs(record, *CONDITION)
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_tinnitus_data()
