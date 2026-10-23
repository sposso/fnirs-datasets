"""
Hu, Y., Liu, Y., Hou, Y. et al. Dataset of natural conversations about appearance using fNIRS.
Sci Data 12, 1486 (2025). https://doi.org/10.1038/s41597-025-05574-9

31 participants, one NIRx Aurora SNIRF recording each (40 channels, 760/850 nm, 7.63 Hz).
Optode positions are NIRx's ICBM152 (MNI) template, identical for every participant.
"""

from pathlib import Path

from .utils import DATA_ROOT, print_summary, snirf_to_record


def load_appearance_data(main_dir=DATA_ROOT / "appearance_dataset", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).rglob("*.snirf")):
        record = snirf_to_record(path)
        record["subject_id"] = path.parent.name  # one folder per participant
        outputs[path.stem] = record

    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_appearance_data()
