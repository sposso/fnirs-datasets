"""
Cheng, S., Yang, X., Ding, Y. (2026). Authority reliance vs. deliberative assessment in processing online rumors:
evidence from fNIRS. Frontiers in Human Neuroscience 20, 1750499. doi:10.3389/fnhum.2026.1750499.
Data: Science Data Bank doi:10.57760/sciencedb.35305 (V1, CC BY 4.0, "raw fNIRs data for in processing online
rumors"); only the SNIRF files and the session description/header files were downloaded.

42 adults, one recording of 24-28 min each: judging whether 16 news items were true or false, without a source label
and then labeled as fact-checked by an authoritative (People's Daily) or a non-authoritative (Jiupai News) source.
NIRSport2 (NIRx; LED sources, 760/850 nm), 10.17 Hz, 8 sources and 8 detectors over the bilateral prefrontal cortex:
23 channels (30-46 mm as stored); all are kept (the paper excludes no participants or channels; it only leaves six
channels out of its regions of interest).
The dataset also holds byte-identical copies of four 2024-03-06 sessions in top-level folders; only the date folders
are read.
Optode positions are NIRx's ICBM152 template stored in the files (their 300 10-5 landmarks equal the NIRx ICBM152
table), i.e. MNI coordinates, identical for every participant; kept as stored, as for the Stroop dataset.
"""

import json
from pathlib import Path

from .utils import DATA_ROOT, print_summary, snirf_to_record


def load_rumor_data(main_dir=DATA_ROOT / "rumor_judgment", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("????-??-??/*/*.snirf")):     # date folder / session folder / file
        record = snirf_to_record(path)
        description = json.loads(path.with_name(f"{path.stem}_description.json").read_text())
        record["subject_id"] = f"sub-{description['subject']}"
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_rumor_data()
