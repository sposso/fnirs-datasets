"""
Chen, Z., Gao, C., Li, T., Ji, X., Liu, S., Xiao, M. (2023). Open access dataset integrating EEG and fNIRS during
Stroop tasks. Sci. Data 10, 618.
Data: figshare collection doi:10.6084/m9.figshare.c.6623431 (fNIRS-EEG_Stroop.zip, CC0), read without extracting it
(the raw fNIRS files are in the inner zip "Raw fNIRS Data In mat.zip").

21 adults performing a Chinese color-word Stroop task (two neutral and two incongruent 16-s blocks) with simultaneous
EEG: one recording of ~4 min each. In-house CW system, 785/850 nm, 100 Hz, 4 sources and 16 detectors: 20 channels
at 3 cm over the frontal cortex; all kept. Raw light intensity: cell "yy", columns 4-43 = 20 channels x (785, 850 nm).
The description's "odd-numbered columns" is ambiguous; 785 nm first is the order whose HbO/HbR match the authors'
preprocessed data (r = 0.53-0.85 for 4 participants, against 0.22-0.52 for the other order). Each column ends with a
zero sample, removed by the shared missing-data rule (fill_or_drop).
Rejected for the main set: no positions (Fig. 2 is a 2D drawing; no coordinate file); channel_xyz is None.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

from loaders.utils import DATA_ROOT, fill_or_drop, intensity_to_raw, print_summary, raw_to_record

SFREQ, SD_DISTANCE = 100.0, 0.03       # Hz, metres (paper)


def load_stroop_eeg_data(main_dir=DATA_ROOT / "stroop_eeg", verbose=True):
    """Return ``{recording_name: record}`` for every raw fNIRS file in the zip file of ``main_dir``."""
    outputs = {}
    with zipfile.ZipFile(Path(main_dir) / "fNIRS-EEG_Stroop.zip") as z:
        inner = "fNIRS-EEG_Stroop/Raw Data/Raw fNIRS Data/Raw fNIRS Data In mat.zip"
        with zipfile.ZipFile(io.BytesIO(z.read(inner))) as raw_zip:
            for member in sorted(n for n in raw_zip.namelist() if n.endswith(".mat")):
                yy = sio.loadmat(io.BytesIO(raw_zip.read(member)), squeeze_me=True)["yy"]
                columns = np.array([np.asarray(yy[i], float).ravel() for i in range(3, 43)])  # 785, 850 per channel
                raw = intensity_to_raw(np.stack([columns[0::2], columns[1::2]]), SFREQ, (785, 850), SD_DISTANCE)
                raw = fill_or_drop(raw, max_gap=round(SFREQ), max_edge=round(60 * SFREQ))  # 1-s gaps, 1-min edges
                record = raw_to_record(raw, allow_2d=True)
                record["channel_xyz"] = None
                record["subject_id"] = Path(member).stem                            # S1 ... S21
                record["coord_frame"] = "none (2D drawing in the paper only)"
                outputs[record["subject_id"]] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_stroop_eeg_data()
