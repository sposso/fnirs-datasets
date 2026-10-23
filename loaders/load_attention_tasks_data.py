"""
Ding, L., Zhang, Y., Xie, Y., He, Y., Wang, Y., Lu, J., Pang, R., Wang, W., Chen, Z. (2025). Cortical activation and
functional connectivity during different attention tasks using functional near-infrared spectroscopy in middle-aged
and elderly people. European Journal of Medical Research 30, 364. doi:10.1186/s40001-025-02597-1.
Data: Mendeley Data 7tcrj59w2p v1, "The characteristics of cortical activation and functional connectivity during the
performance of different attention tasks in brain-injured patients: A fNIRS study" (doi:10.17632/7tcrj59w2p.1,
CC BY 4.0); DATA.zip (SHA256 checked) is read directly, without extracting it.

Patients with brain injury and healthy middle-aged and elderly adults performed four attention tasks, one Homer2 .nirs
recording each: digit cancellation (DCT), paced auditory serial addition (PASAT), Stroop colour-word and trail making
(TMT), as blocks of 30-s task and 60-s rest (PASAT: 150 s and 180 s). 325 recordings of 4.8-6.2 min (29.5 h) from 87
participants. The paper describes the healthy group only.
NirSmart (Danyang Huichuang, China; 730/850 nm), 11 Hz: 24 sources and 16 detectors forming 48 channels over the
frontal, temporal, parietal and occipital lobes, all kept. The files hold raw intensity and a 2D probe design in which
every source-detector separation is 30 mm; Beer-Lambert uses these distances.
Positions: mni.mat (five identical copies; also Channel.mat, NIRS-SPM standalone registration on the SPM T1 template)
gives each channel's MNI coordinates in the channel order of the files; they are used as given.
Participants: the folder names hold names, sex and birth date; a participant is identified across the four task
folders by sex, birth date and the first two characters of the name (the task words added to the names differ between
folders), and subject_id is an anonymous hash of these.
"""

import hashlib
import io
import re
import zipfile
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

TASKS = {"DCT": "DCT", "PASAT": "PASAT", "Stoop": "Stroop", "TMT": "TMT"}   # folder -> task name ("Stoop" sic)
MARKERS = ["持续", "分配", "选择", "转移", "健康", "FP", "ZY", "丁", "（", "）", "(", ")", "_", " "]  # added to names


def member_name(info):
    """Zip member name; names without the UTF-8 flag are GBK-encoded (Chinese Windows)."""
    return info.filename if info.flag_bits & 0x800 else info.filename.encode("cp437").decode("gbk")


def subject_id(folder):
    """Anonymous participant ID from a folder name '<name><task word>_<sex>_<birth date>_'."""
    name, sex, birth = re.search(r"^(.*?)_?(男|女)_(\d{4}-\d{2}-\d{2})", folder).groups()
    for marker in MARKERS:
        name = name.replace(marker, "")
    return "sub-" + hashlib.sha1(f"{sex}|{birth}|{name[:2]}".encode()).hexdigest()[:8]


def read_nirs(data):
    """Raw intensity of one Homer2 .nirs file (bytes) as an MNE Raw, optodes at the 2D probe design (z = 0, metres)."""
    nirs = loadmat(io.BytesIO(data), squeeze_me=True, struct_as_record=False)
    sd = nirs["SD"]
    meas = np.asarray(sd.MeasList).astype(int)                                 # source, detector, -, wavelength index
    wavelengths = np.ravel(sd.Lambda)
    order = np.lexsort((meas[:, 3], meas[:, 1], meas[:, 0]))                  # wavelengths of a pair together
    names = [f"S{meas[i, 0]}_D{meas[i, 1]} {wavelengths[meas[i, 3] - 1]}" for i in order]
    t = np.asarray(nirs["t"], float)
    info = mne.create_info(names, 1.0 / np.median(np.diff(t)), "fnirs_cw_amplitude")
    src = np.c_[np.asarray(sd.SrcPos, float)[:, :2], np.zeros(len(sd.SrcPos))] / 100.0   # cm -> m
    det = np.c_[np.asarray(sd.DetPos, float)[:, :2], np.zeros(len(sd.DetPos))] / 100.0
    for ch, i in zip(info["chs"], order):
        ch["loc"][3:6], ch["loc"][6:9] = src[meas[i, 0] - 1], det[meas[i, 1] - 1]
        ch["loc"][:3] = (ch["loc"][3:6] + ch["loc"][6:9]) / 2
        ch["loc"][9] = wavelengths[meas[i, 3] - 1]
    raw = mne.io.RawArray(np.asarray(nirs["d"], float)[:, order].T, info, verbose=False)
    first_wavelength = meas[meas[:, 3] == 1]                                   # channel order of mni.mat
    return raw, [f"S{s}_D{d}" for s, d in first_wavelength[:, :2]]


def load_attention_tasks_data(main_dir=DATA_ROOT / "attention_tasks", verbose=True):
    """Return ``{recording_name: record}`` for every .nirs file in ``main_dir``/DATA.zip."""
    outputs = {}
    with zipfile.ZipFile(Path(main_dir) / "DATA.zip") as archive:
        mni = loadmat(io.BytesIO(archive.read("DCT/mni.mat")), squeeze_me=True)["mni"] / 1000.0  # 48 x 3, mm -> m
        for info in archive.infolist():
            name = member_name(info)
            if not name.lower().endswith(".nirs"):
                continue
            parts = name.split("/")
            raw, pairs = read_nirs(archive.read(info))
            sfreq = raw.info["sfreq"]
            raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
            record = raw_to_record(raw, channel_pos=dict(zip(pairs, mni)))
            record["subject_id"] = subject_id(parts[-2])
            outputs[f"{TASKS[parts[0]]}_{record['subject_id']}"] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_attention_tasks_data()
