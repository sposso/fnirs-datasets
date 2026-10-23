"""
Shared reader for the University of Michigan Language and Literacy Lab (Kovelman) datasets recorded with a TechEn CW6
(690/830 nm) and the lab's bilateral 46-channel cap: story listening (load_alice_data), morphological and phonological
awareness (load_morphology_data) and sentence prosody (load_prosody_data).

The .nirs files (Homer format, MATLAB v5 or v7.3) store raw light intensity and only a flat 2D drawing of the probe
(z = 0). The cap's 3D channel positions come from CHMNI_Bilateral46_AUG2020.mat, shipped with the prosody dataset:
46 MNI points (mm) on the cortex, one per channel, 1-23 left and 24-46 right (Pasquinelli et al. 2023). Their order is
the order of the 46 source-detector pairs in the files' measurement list (PAIRS below). This order is not documented
with the file; it was inferred: (1) within each hemisphere, the distances between the MNI points and between the
pairs' positions in the 2D drawing have a rank correlation of 0.986 (10,000 random orders: at most 0.52); (2) the regions
of the paper's channel tables match the MNI points.
Some files hold 8 more pairs (sources 7, 8, 15, 16): they have no MNI point and are dropped. The 46 shared pairs have
identical 2D positions in all three datasets.
Channels switched off in the file (SD.MeasListAct = 0, the authors' exclusions) are dropped.
Beer–Lambert uses 27 mm for every channel ("~2.7 cm apart", Sun et al. 2022): the files give no real distance.
"""

from pathlib import Path

import h5py
import mne
import numpy as np
import scipy.io as sio

from .utils import DATA_ROOT

CHMNI = DATA_ROOT / "prosody/CHMNI_Bilateral46_AUG2020.mat"  # 46 channel MNI points (mm), from the prosody dataset
SD_DISTANCE = 0.027                                     # m, source-detector distance used by Beer–Lambert
WAVELENGTHS = (690, 830)
PAIRS = [  # the cap's 46 channels, in the files' measurement order = the rows of CHMNI
    (1, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 7), (4, 8), (4, 9), (4, 10), (5, 5), (5, 7), (5, 11), (6, 7), (6, 9), (6, 11), (6, 12),
    (9, 17), (9, 18), (9, 19), (9, 20), (10, 19), (10, 20), (10, 21), (10, 22), (11, 21), (11, 22), (11, 23),
    (11, 24), (12, 23), (12, 24), (12, 25), (12, 26), (13, 21), (13, 23), (13, 27), (14, 23), (14, 25),
    (14, 27), (14, 28),
]


def channel_mni():
    """``{"S1_D1": MNI xyz in metres}`` for the cap's 46 channels."""
    xyz = sio.loadmat(CHMNI, squeeze_me=True)["CHMNI"].astype(float) / 1000.0
    return {f"S{s}_D{d}": xyz[i] for i, (s, d) in enumerate(PAIRS)}


def read_nirs(path):
    """Intensity (samples x measurements), sampling rate, measurement list and active flags of a .nirs file."""
    try:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        sd, t, d = m["SD"], np.ravel(m["t"]), np.asarray(m["d"], float)
        ml = np.asarray(sd.MeasList)
        act = np.ravel(getattr(sd, "MeasListAct", np.ones(len(ml))))
    except NotImplementedError:                                       # MATLAB v7.3 (HDF5)
        with h5py.File(path, "r") as f:
            t, d = np.ravel(f["t"][()]), np.array(f["d"][()], float)
            ml = np.array(f["SD/MeasList"][()]).T
            act = np.ravel(f["SD/MeasListAct"][()]) if "SD/MeasListAct" in f else np.ones(len(ml))
        d = d.T if d.shape[0] != len(t) else d
    if d.shape[0] != len(t):
        raise ValueError(f"{path}: {d.shape[0]} samples but {len(t)} time points")
    return d, 1.0 / np.median(np.diff(t)), ml.astype(int), act


def read_raw(path, mni=None):
    """Raw intensity of the cap's 46 channels (minus those switched off) as an MNE Raw, each channel at its MNI
    point (``mni``, by default the CHMNI points), with source and detector 27 mm apart around it."""
    d, sfreq, ml, act = read_nirs(path)
    mni = channel_mni() if mni is None else mni                         # {"S1_D1": MNI xyz in metres}
    off = {(s, det) for (s, det, _, _), a in zip(ml, act) if a == 0}   # pair off if either wavelength is off
    names, columns = [], []
    for s, det in PAIRS:
        if (s, det) in off:
            continue
        for w, wl in enumerate(WAVELENGTHS, start=1):
            columns.append(np.flatnonzero((ml[:, 0] == s) & (ml[:, 1] == det) & (ml[:, 3] == w))[0])
            names.append(f"S{s}_D{det} {wl}")
    info = mne.create_info(names, sfreq, "fnirs_cw_amplitude")
    for ch in info["chs"]:
        xyz = mni[ch["ch_name"].split(" ")[0]]
        ch["loc"][:3] = xyz
        ch["loc"][3:6] = xyz - [SD_DISTANCE / 2, 0, 0]
        ch["loc"][6:9] = xyz + [SD_DISTANCE / 2, 0, 0]
        ch["loc"][9] = float(ch["ch_name"].split(" ")[1])
    return mne.io.RawArray(d[:, columns].T, info, verbose=False)


def nirs_files(main_dir):
    """All .nirs files below ``main_dir``, sorted."""
    return sorted(Path(main_dir).rglob("*.nirs"))
