"""
Ding, Y., Deng, Z., Zhang, W., Lyu, W., Duffy, B. M., Zhang, D. (2026). Wired to believe: source-authority cues shape
perceived AI capability and neural persuasion. Ergonomics, 1-18. doi:10.1080/00140139.2026.2734930.
Data: Science Data Bank doi:10.57760/sciencedb.42071 (V1, CC BY-SA 4.0, "fNIRS experimental data"), folder
三组原始数据集 ("three groups of raw data"); only the SNIRF files and the session description/header files were
downloaded.

72 adults in three groups of 24 (folders H_, M_ and L_), one recording of 3-6 min each. NIRSport2 (NIRx; LED sources,
760/850 nm), 10.17 Hz, the same 8-source/8-detector prefrontal montage as the online-rumor dataset of the same lab:
23 channels (30-46 mm as stored); all are kept.
Each folder is one participant (subject_id = folder name): the description files of M_sub2 and H_sub2 both carry the
subject number 015, but they are different recordings in different groups.
Optode positions are NIRx's ICBM152 template stored in the files (their 300 10-5 landmarks equal the NIRx ICBM152
table), i.e. MNI coordinates, identical for every participant; kept as stored.
"""

from pathlib import Path

from .utils import DATA_ROOT, print_summary, snirf_to_record


def load_ai_persuasion_data(main_dir=DATA_ROOT / "ai_persuasion", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("*/[HML]_sub*/*.snirf")):       # group folder / participant folder / file
        record = snirf_to_record(path)
        record["subject_id"] = path.parent.name                             # e.g. H_sub2
        outputs[f"{path.parent.name}_{path.stem}"] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_ai_persuasion_data()
