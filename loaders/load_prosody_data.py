"""
Pasquinelli, R., Tessier, A. M., Karas, Z., Hu, X., Kovelman, I. (2023). The development of left hemisphere
lateralization for sentence-level prosodic processing. Journal of Speech, Language, and Hearing Research 66(4),
1365-1377. doi:10.1044/2022_JSLHR-22-00103. Data: Deep Blue, doi:10.7302/xj95-gv44 (raw .nirs files only).

Normal-hearing adults (NHA) and children (NHC), and children with cochlear implants (CI), performed a sentence
intonation task, a rhyming task and an auditory oddball task. TechEn CW6, 690/830 nm, 50 Hz.
Channels and positions: the lab's 46-channel cap, see kovelman_cap.py; the CHMNI file of 3D positions comes with this
dataset. Pairs switched off in the file are dropped.
"""

from pathlib import Path

from .kovelman_cap import nirs_files, read_raw
from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record


def load_prosody_data(main_dir=DATA_ROOT / "prosody/fNIRS data", verbose=True):
    """Return ``{recording_name: record}`` for every raw .nirs file in ``main_dir``."""
    outputs = {}
    for path in nirs_files(main_dir):
        if "Raw" not in path.parts:
            continue
        raw = read_raw(path)
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw)
        group, task, number = path.stem.split("_")                       # e.g. NHA, Intonation, CCI005
        record["subject_id"] = f"{group}_{number}"
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_prosody_data()
