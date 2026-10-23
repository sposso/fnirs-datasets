"""
Ji, C., Bao, Y., Li, X., Pu, W., Lin, J., Zhang, Y., Chen, Y., Li, T. (2024). EEG and fNIRS datasets based on Stroop
task during two weeks of high-altitude exposure in new immigrants. Sci. Data 11, 350.
Data: figshare collection doi:10.6084/m9.figshare.c.6979581 (CC0), read from its zip without extracting it.

80 young men, 40 after two weeks at 4,500 m (High_altitude) and 40 at 800 m (Low_altitude), performing a Stroop task
(120 trials of 2 s) with simultaneous EEG: one 240-s recording each (task only). Four channels "evenly distributed
around the frontal center", 5 Hz. Only HbO/HbR computed with the modified Beer-Lambert law (785/850 nm) are shared,
in umol/L (the unit of the paper's results): multiplied by 1e-6 to get M. The source-detector distance is not
documented, so sd_distances are NaN. Isolated non-finite samples (13 files) follow the shared missing-data rule
(fill_or_drop, non-finite only, as Hb can be negative).
The two groups reuse subject numbers, so each group folder is treated independently.
Rejected for the main set: no positions (none in the files or the paper); channel_xyz is None.
"""

import io
import re
import zipfile
from pathlib import Path

import mne
import numpy as np
import scipy.io as sio

from loaders.utils import DATA_ROOT, fill_or_drop, hb_to_record, print_summary

SFREQ = 5.0


def load_immigrants_data(main_dir=DATA_ROOT / "immigrants", verbose=True):
    """Return ``{recording_name: record}`` for every participant in the zip file of ``main_dir``."""
    outputs = {}
    with zipfile.ZipFile(next(Path(main_dir).glob("*.zip"))) as z:
        for member in sorted(n for n in z.namelist() if "/fNIRS_data/" in n and "/HbO2_subject" in n):
            hbo = sio.loadmat(io.BytesIO(z.read(member)), squeeze_me=True)["HbO2"].T          # (4, 1200), umol/L
            hbr = sio.loadmat(io.BytesIO(z.read(member.replace("HbO2_", "Hb_"))), squeeze_me=True)["Hb"].T
            names = [f"CH{i + 1} {kind}" for i in range(4) for kind in ("hbo", "hbr")]
            raw = mne.io.RawArray(np.stack([hbo, hbr], axis=1).reshape(8, -1) * 1e-6,
                                  mne.create_info(names, SFREQ, ["hbo", "hbr"] * 4), verbose=False)
            raw = fill_or_drop(raw, max_gap=round(SFREQ), max_edge=round(60 * SFREQ), positive=False)
            x = raw.get_data()
            record = hb_to_record(x[0::2], x[1::2], SFREQ, None, np.full(len(x) // 2, np.nan))
            group = member.split("/")[-2].split("_")[0]                            # High or Low
            number = re.search(r"\((\d+)\)", member).group(1)
            record["subject_id"] = f"{group}_{number}"                             # e.g. High_6
            record["coord_frame"] = "none (four frontal channels, no positions)"
            outputs[record["subject_id"]] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_immigrants_data()
