"""
Huang, R. (2024). 20231211 social distance&gain-loss to deceived [dataset]. Science Data Bank,
doi:10.57760/sciencedb.14965 (CC BY-NC 4.0); no paper found.

Hyperscanning of 78 sender (a) - receiver (b) dyads in a deception task under social distance and gain or loss
framing: 552 recordings of ~11 min, read from fNIRS_acquirement.rar without extracting it. File names: role, dyad,
g(ain) or l(oss), run (dyads 66-88 have one run per condition). Shimadzu device, 8 x 2 holder, 22 channels per
participant, 13.3 Hz; all kept. Only the HbO/HbR of the Shimadzu software are shared (mM*cm). The source-detector
distance is not documented: 30 mm (the Shimadzu holder spacing of the other datasets) is assumed, i.e. division by
18 cm (30 mm x partial pathlength factor 6) to get M.
Rejected for the main set: no positions (no probe or position file; the shared conversion script expects a probe
file made by the dataset user); channel_xyz is None.
"""

import re
from pathlib import Path

import numpy as np
import rarfile

from loaders.utils import DATA_ROOT, PPF, hb_to_record, print_summary, read_shimadzu_txt

SD_DISTANCE = 0.03                              # metres (assumed, not documented)
TO_MOLAR = 1e-3 / (SD_DISTANCE * 100 * PPF)    # mM*cm -> M, dividing by 18 cm


def load_deception_data(main_dir=DATA_ROOT / "deception", verbose=True):
    """Return ``{recording_name: record}`` for every TXT file of ``fNIRS_acquirement.rar`` in ``main_dir``."""
    outputs = {}
    with rarfile.RarFile(Path(main_dir) / "fNIRS_acquirement.rar") as rar:
        for member in sorted(n for n in rar.namelist() if n.upper().endswith(".TXT")):
            pairs, sfreq, hb = read_shimadzu_txt(rar.read(member).decode(errors="ignore"))   # oxy, deoxy, total
            record = hb_to_record(hb[0::3] * TO_MOLAR, hb[1::3] * TO_MOLAR, sfreq, None,
                                  np.full(len(pairs), SD_DISTANCE))
            stem = Path(member).stem.lower()                                     # e.g. a10g1, b83l
            record["subject_id"] = re.match(r"[ab]\d+", stem).group()            # a10 = sender of dyad 10
            record["coord_frame"] = "none (source-detector pairs only)"
            outputs[stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_deception_data()
