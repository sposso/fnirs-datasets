"""
Chen, R., Yu, L., Jin, Y., Li, P. (2025). The Impact of Masking Stimuli on Audiovisual Integration Mechanisms
[dataset]. Science Data Bank, doi:10.57760/sciencedb.21750 (CC BY-NC-ND 4.0); no paper found.

25 participants performing audiovisual tasks with visual and auditory masking (16 blocks): one recording of ~24 min
each. LABNIRS (Shimadzu; 780/805/830 nm), 9.26 Hz, 48 channels; all kept. Only the HbO/HbR of the Shimadzu software
are shared (mM*cm). The source-detector distance is not documented: 30 mm (the Shimadzu holder spacing of the other
datasets) is assumed, i.e. division by 18 cm (30 mm x partial pathlength factor 6) to get M.
File names are participants' initials: subject_id is an anonymous number (sub-01 ... in file order).
Rejected for the main set: no positions (no probe or position file; the files list only source-detector pairs);
channel_xyz is None.
"""

from pathlib import Path

import numpy as np

from loaders.utils import DATA_ROOT, PPF, hb_to_record, print_summary, read_shimadzu_txt

SD_DISTANCE = 0.03                              # metres (assumed, not documented)
TO_MOLAR = 1e-3 / (SD_DISTANCE * 100 * PPF)    # mM*cm -> M, dividing by 18 cm


def load_av_masking_data(main_dir=DATA_ROOT / "av_masking", verbose=True):
    """Return ``{recording_name: record}`` for every raw TXT file in ``main_dir``."""
    outputs = {}
    paths = sorted(Path(main_dir).glob("extracted/data/fNIRS data/raw data/*.TXT"))
    for number, path in enumerate(paths, start=1):
        pairs, sfreq, hb = read_shimadzu_txt(path.read_text(errors="ignore"))  # per channel: oxy, deoxy, total
        record = hb_to_record(hb[0::3] * TO_MOLAR, hb[1::3] * TO_MOLAR, sfreq, None, np.full(len(pairs), SD_DISTANCE))
        record["subject_id"] = f"sub-{number:02d}"
        record["coord_frame"] = "none (source-detector pairs only)"
        outputs[record["subject_id"]] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_av_masking_data()
