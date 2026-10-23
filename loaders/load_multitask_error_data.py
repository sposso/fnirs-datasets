"""
Zhu, S., Guo, X., Song, X., Chen, S., He, Y. (2025). The Neurophysiological Mechanisms of Post-Error Responses in
Multitasking Scenarios [dataset]. Science Data Bank, doi:10.57760/sciencedb.28380 (V1, CC BY 4.0); no paper found.
Only the .nirs, probeInfo.mat, description.json and digpts.txt files of the 31 sessions (近红外数据) were downloaded.

31 adults performing a multitasking task (December 2022 and March 2023), one recording of 37-58 min each.
NIRSport (NIRx; 760/850 nm), 4.36 Hz, 16 sources and 16 detectors on standard 10-5 positions: 39 channels over the
bilateral dorsolateral prefrontal, frontopolar, orbitofrontal and parietal (angular gyrus) cortex; all are kept.
The folders hold no NIRStar .hdr, so the raw intensities are read from the Homer .nirs file of each session.
Each session folder is one participant (subject_id = folder name): two sessions (2022-12-11_001 and _002) carry the
same subject number 09 in description.json, while the dataset reports 31 participants.
Optode positions: each optode is named by its 10-5 position in probeInfo.mat, and digpts.txt equals the NIRx ICBM152
table for those labels, i.e. the template, identical for every participant; each optode gets the NIRx ICBM152 MNI
position of its label (as in Shin 2018). The scalp-localization folder (头皮定位数据) holds a single example
digitization and is not used.
"""

import json
from pathlib import Path

import mne
import numpy as np
import scipy.io as sio

from .utils import DATA_ROOT, NIRX_ICBM152, print_summary, raw_to_record


def read_session(nirs_path):
    """Raw intensity of one Homer .nirs file as an MNE Raw, optodes at the NIRx ICBM152 positions of their labels."""
    nirs = sio.loadmat(nirs_path, squeeze_me=True, struct_as_record=False)
    sd, data = nirs["SD"], np.asarray(nirs["d"], float).T            # (n_measurements, n_times)
    sfreq = 1.0 / np.median(np.diff(nirs["t"]))
    measurements = np.asarray(sd.MeasList).astype(int)                # source, detector, -, wavelength index
    order = np.lexsort((measurements[:, 3], measurements[:, 1], measurements[:, 0]))  # pairs adjacent, as MNE expects
    names = [f"S{s}_D{d} {int(sd.Lambda[w - 1])}" for s, d, _, w in measurements[order]]
    info = mne.create_info(names, sfreq, "fnirs_cw_amplitude")
    probe = sio.loadmat(next(nirs_path.parent.glob("*_probeInfo.mat")), squeeze_me=True, struct_as_record=False)
    labels = {f"S{i + 1}": label for i, label in enumerate(probe["probeInfo"].probes.labels_s)}
    labels.update({f"D{i + 1}": label for i, label in enumerate(probe["probeInfo"].probes.labels_d)})
    template = json.loads(NIRX_ICBM152.read_text())                   # 10-5 label -> MNI xyz (mm)
    for ch in info["chs"]:
        pair, wavelength = ch["ch_name"].split(" ")
        source, detector = pair.split("_")
        ch["loc"][3:6] = np.array(template[labels[source]]) / 1000.0
        ch["loc"][6:9] = np.array(template[labels[detector]]) / 1000.0
        ch["loc"][:3] = (ch["loc"][3:6] + ch["loc"][6:9]) / 2
        ch["loc"][9] = float(wavelength)
    return mne.io.RawArray(data[order], info, verbose=False)


def load_multitask_error_data(main_dir=DATA_ROOT / "multitask_error", verbose=True):
    """Return ``{recording_name: record}`` for every session folder in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("近红外数据/*/*.nirs")):
        record = raw_to_record(read_session(path))
        record["subject_id"] = path.parent.name                         # e.g. 2022-12-10_004
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_multitask_error_data()
