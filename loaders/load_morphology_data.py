"""
Sun, X., Zhang, K., Marks, R., Karas, Z., Eggleston, R., Nickerson, N., Yu, C.-L., Wagley, N., Hu, X., Caruso, V.,
Chou, T.-L., Satterfield, T., Tardif, T., Kovelman, I. (2022). Morphological and phonological processing in English
monolingual, Chinese-English bilingual, and Spanish-English bilingual children: An fNIRS neuroimaging dataset.
Data in Brief 42, 108048. doi:10.1016/j.dib.2022.108048. Data: Deep Blue, updated version (ft848r79k, doi:10.7302/gg8n-mq88).

Children aged 5-11 (English monolingual, Chinese-English and Spanish-English bilingual) performed morphological and
phonological awareness tasks (7.2 min) in each of their languages. TechEn CW6, 690/830 nm, 50 Hz (68 of the loaded files at 25 Hz).
Channels and positions: the lab's 46-channel cap, see kovelman_cap.py (most files hold 54 pairs; the 8 without an
MNI point are dropped, as are pairs switched off in the file).
Skipped: 3089_EN_PA (truncated file) and 2072_EN_MA, 3027_EN_PA (time vector inconsistent with the data).
"""

from .kovelman_cap import nirs_files, read_raw
from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

SKIPPED = {"3089_EN_PA", "2072_EN_MA", "3027_EN_PA"}


def load_morphology_data(main_dir=DATA_ROOT / "morphology", verbose=True):
    """Return ``{recording_name: record}`` for every .nirs file in ``main_dir``."""
    outputs = {}
    for path in nirs_files(main_dir):
        if path.stem in SKIPPED:
            continue
        raw = read_raw(path)
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw)
        record["subject_id"] = path.stem.split("_")[0]                   # e.g. 1001 (same IDs as the Alice dataset)
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_morphology_data()
