"""
Zeng, L. (2025). fNIRS dataset of Stroop task with block design [dataset]. Science Data Bank,
doi:10.57760/sciencedb.28229 (CC BY-NC 4.0); no paper found (same laboratory as the Stroop decoding dataset).

45 participants (20 in a decoded-neurofeedback group, 25 in a sham group) performing a block-design Chinese color-word
Stroop task (30 blocks, ~25 min); one session per participant is shared (stroop1), read from 06_raw_nirs.rar without
extracting it. The two groups reuse subject numbers, so each group folder is treated independently.
LABNIRS (Shimadzu), 25.6 Hz, 35 channels at 30 mm; all kept (no channel flags in the files). Only the HbO/HbR of
the Shimadzu software are shared (mM*cm): divided by 18 cm (30 mm x partial pathlength factor 6) to get M.
Rejected for the main set: no 3D positions. The probe file SD2329-2.SD (the same as the Stroop decoding dataset) is a
flat 2D grid in mm (z = 0; stored as 8-bit integers); channel positions are the midpoints of its optodes, kept as
provided.
"""

import io
from pathlib import Path

import numpy as np
import rarfile
import scipy.io as sio

from loaders.utils import DATA_ROOT, PPF, hb_to_record, print_summary, read_shimadzu_txt

SD_DISTANCE = 0.03                              # metres (probe file)
TO_MOLAR = 1e-3 / (SD_DISTANCE * 100 * PPF)    # mM*cm -> M, dividing by 18 cm


def load_stroop_block_data(main_dir=DATA_ROOT / "stroop_block", verbose=True):
    """Return ``{recording_name: record}`` for every TXT file of ``06_raw_nirs.rar`` in ``main_dir``."""
    outputs = {}
    with rarfile.RarFile(Path(main_dir) / "06_raw_nirs.rar") as rar:
        probe = sio.loadmat(io.BytesIO(rar.read("06 raw_nirs/SD2329-2.SD")), squeeze_me=True,
                            struct_as_record=False)["SD"]
        sources, detectors = probe.SrcPos.astype(float) / 1000, probe.DetPos.astype(float) / 1000   # mm -> m
        for member in sorted(n for n in rar.namelist() if n.upper().endswith(".TXT")):
            pairs, sfreq, hb = read_shimadzu_txt(rar.read(member).decode(errors="ignore"))   # oxy, deoxy, total
            xyz = np.array([(sources[s - 1] + detectors[d - 1]) / 2 for s, d in pairs])
            record = hb_to_record(hb[0::3] * TO_MOLAR, hb[1::3] * TO_MOLAR, sfreq, xyz,
                                  np.full(len(pairs), SD_DISTANCE))
            group = member.split("/")[1].split()[0]                              # decnef or sham
            record["subject_id"] = f"{group}_{Path(member).stem}"                 # e.g. sham_sub09
            record["coord_frame"] = "flat 2D probe grid (z = 0) of SD2329-2.SD, mm"
            outputs[record["subject_id"]] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_stroop_block_data()
