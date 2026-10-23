"""
Sztuka, I. M., Kühn, S. (2025). Neurocognitive dynamics and behavioral differences of symmetry and asymmetry
processing in working memory: insights from fNIRS. Scientific Reports 15, 4740. doi:10.1038/s41598-024-84988-8.
Data: OSF z7kw3 ("Symmetry and Visual Working Memory", doi:10.17605/OSF.IO/Z7KW3, CC BY 4.0), rawdata.zip (BIDS).

37 adults, one recording of 47-57 min each: 5 min of eyes-closed rest, then a visual working-memory task with
symmetrical and asymmetrical stimuli. Two NIRSport2 devices (LED sources, 760/850 nm, APD detectors), 5.09 Hz:
32 sources and 32 detectors plus 16 short detectors, 100 long channels (30 mm) and 16 short channels (8 mm) over the
whole head; all are kept (all "good" in channels.tsv).
Excluded: participants flagged by the authors in participants.csv (included_nirs = no: 11 of 37, more than 20% of the
channels with poor signal), leaving 26.
Optode positions are NIRx's template montage (NIRSite) stored in CapTrak (BIDS coordsystem.json), identical for every
participant; ``head_to_mni`` converts them to MNI, as for FRESH Audio.
"""

import csv
from pathlib import Path

from .utils import DATA_ROOT, head_to_mni, print_summary, snirf_to_record


def load_symmetry_wm_data(main_dir=DATA_ROOT / "symmetry_wm", verbose=True):
    """Return ``{recording_name: record}`` for every included SNIRF file in ``main_dir``."""
    with open(Path(main_dir) / "participants.csv", encoding="utf-8-sig") as f:
        included = {row["participant_id"] for row in csv.DictReader(f) if row["included_nirs"] == "yes"}
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*_nirs.snirf")):
        subject = path.name.split("_")[0]                                   # e.g. sub-06
        if subject not in included:
            continue
        record = snirf_to_record(path)
        record["channel_xyz"] = head_to_mni(record["channel_xyz"])          # file positions are CapTrak
        record["subject_id"] = subject
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_symmetry_wm_data()
