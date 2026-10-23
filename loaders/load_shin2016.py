"""
Shin, J., von Lühmann, A., Blankertz, B., Kim, D.-W., Jeong, J., Hwang, H.-J., Müller, K.-R. (2017). Open access
dataset for EEG+NIRS single-trial classification. IEEE Transactions on Neural Systems and Rehabilitation Engineering
25(10), 1735-1745. doi:10.1109/TNSRE.2016.2628057.
Data: https://doc.ml.tu-berlin.de/hBCI/ (NIRS_01-29.zip; BBCI toolbox .mat files).

29 adults, 6 sessions each of ~12 min: left/right-hand motor imagery (sessions 1, 3, 5) and mental arithmetic vs.
baseline (sessions 2, 4, 6); 174 recordings. The separately recorded motion-artifact sessions (cnt_artifact) are not
used. NIRScout (760/850 nm), 14 sources and 16 detectors forming 36 channels (30 mm) over the prefrontal, motor and
occipital cortex; the files hold raw light intensity at 10 Hz. All channels are kept (no channel quality information
was published); the few zero/negative samples are handled by utils.fill_or_drop.
Positions: every optode is labeled with its 10-5 position (mnt.source/detector.clab) and placed at the NIRx ICBM152
MNI position of that label, as in Shin 2018.
"""

import json
from pathlib import Path

import mne
import numpy as np
import scipy.io as sio

from .utils import DATA_ROOT, NIRX_ICBM152, fill_or_drop, print_summary, raw_to_record, set_optode_positions

TASKS = ["mi", "ma"] * 3  # cnt{1..6}: motor imagery and mental arithmetic sessions, alternating
WAVELENGTHS = (760, 850)


def read_subject(folder):
    """The 6 recordings of one subject folder as MNE Raws of raw intensity, optodes at their MNI positions."""
    mnt = sio.loadmat(folder / "mnt.mat", squeeze_me=True, struct_as_record=False)["mnt"]
    cnts = sio.loadmat(folder / "cnt.mat", squeeze_me=True, struct_as_record=False)["cnt"]
    table = json.loads(NIRX_ICBM152.read_text())                           # 10-5 label -> MNI (mm)
    optodes = {f"S{i + 1}": np.array(table[l]) / 1000 for i, l in enumerate(mnt.source.clab) if l != "-"}
    optodes |= {f"D{i + 1}": np.array(table[l]) / 1000 for i, l in enumerate(mnt.detector.clab) if l != "-"}
    pairs = [f"S{s}_D{d}" for s, d in mnt.sd]                              # 36 channels, same order as cnt.x
    names = [f"{p} {wl}" for p in pairs for wl in WAVELENGTHS]
    raws = []
    for cnt in cnts:
        n = len(pairs)
        x = np.empty((2 * n, cnt.x.shape[0]))
        x[0::2], x[1::2] = cnt.x[:, :n].T, cnt.x[:, n:].T                 # columns: 36 low-WL then 36 high-WL
        info = mne.create_info(names, float(cnt.fs), "fnirs_cw_amplitude")
        for ch in info["chs"]:
            ch["loc"][9] = float(ch["ch_name"].split(" ")[1])              # wavelength (nm)
        set_optode_positions(info, optodes)
        raws.append(mne.io.RawArray(x, info, verbose=False))
    return raws


def load_shin2016_data(main_dir=DATA_ROOT / "shin2016", verbose=True):
    """Return ``{recording_name: record}`` for every session of every subject in ``main_dir``."""
    outputs = {}
    for folder in sorted(Path(main_dir).glob("subject *")):
        subject = "VP" + folder.name.split()[1]                             # e.g. VP01
        for k, raw in enumerate(read_subject(folder)):
            sfreq = raw.info["sfreq"]
            raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
            if raw is None:
                continue
            record = raw_to_record(raw)
            record["subject_id"] = subject
            outputs[f"{subject}_{TASKS[k]}_{k // 2 + 1}"] = record           # e.g. VP01_mi_1
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_shin2016_data()
