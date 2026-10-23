"""
Schroeder, M. L., Sherafati, A., Ulbrich, R. L., Wheelock, M. D., Svoboda, A. M., Klein, E. D., George, T. G.,
Tripathy, K., Culver, J. P., Eggebrecht, A. T. (2023). Mapping cortical activations underlying covert and overt
language production using high-density diffuse optical tomography. NeuroImage 276, 120190.
doi:10.1016/j.neuroimage.2023.120190.
Data: NITRC NeuroDOT project, "Schroeder et al. 2023 Language Production" (Schroeder_et_al_2023_Language_Production.zip,
read directly from the zip: the disk cannot hold the 37.5 GB unpacked).

33 right-handed native English-speaking adults: silent reading of single nouns (RW), covert verb generation (CV) and
overt verb generation (OV); runs of ~3.6 min, 1-6 sessions per participant (579 runs).
WashU HD-DOT: 96 LED sources and 92 avalanche photodiode detectors (750/850 nm), 10 Hz, 1842 source-detector pairs
with first to fourth nearest-neighbor separations of 13, 30, 39 and 47 mm; only pairs up to 35 mm are kept (856), as
for the other high-density datasets.
Files: an early NeuroDOT ndot2snirf export that MNE cannot read (one measurementList with array fields, strings stored
as character arrays); read with h5py. Raw light intensity.
Positions: identical to NeuroDOT's Pad_Adult_96x92 (0.000 mm), i.e. voxel indices of NeuroDOT's MNI152 volume
(mni152nl_T1: center 99 -99 -117, mmppix 1 -1 -1). NeuroDOT's change_space_coords maps them to MNI as
(99 - x, y - 135, z - 73) mm; the paper uses the nonlinear ICBM152 atlas as head model. The map is a flip plus a
shift, so source-detector distances are unchanged.
Not applied: the paper's exclusion of 11 high-motion runs (not identified in the files).
"""

import io
import zipfile
from pathlib import Path

import h5py
import mne
import numpy as np

from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

ZIP = DATA_ROOT / "language_production/Schroeder_et_al_2023_Language_Production.zip"
MAX_SEPARATION = 0.035  # m, longest source-detector separation kept


def voxel_to_mni(xyz):
    """NeuroDOT MNI152 voxel indices (Pad_Adult_96x92) to MNI in metres (NeuroDOT change_space_coords)."""
    xyz = np.asarray(xyz, float)
    return np.c_[99 - xyz[:, 0], xyz[:, 1] - 135, xyz[:, 2] - 73] / 1000.0


def read_run(data):
    """Raw intensity of one run (bytes of a SNIRF file) as an MNE Raw, each channel at its MNI position."""
    with h5py.File(io.BytesIO(data), "r") as f:
        ml = f["nirs/data1/measurementList1"]
        sources, detectors, wl_index = (np.ravel(ml[k][()]).astype(int) for k in ("sourceIndex", "detectorIndex",
                                                                                     "wavelengthIndex"))
        wavelengths = np.ravel(f["nirs/probe/wavelengths"][()])
        x = f["nirs/data1/dataTimeSeries"][()]                              # samples x measurements
        t = np.ravel(f["nirs/data1/time"][()])
        src = voxel_to_mni(f["nirs/probe/sourcePos3D"][()])
        det = voxel_to_mni(f["nirs/probe/detectorPos3D"][()])
    columns, names = [], []
    pairs = sorted({(s, d) for s, d in zip(sources, detectors)})
    for s, d in pairs:
        if np.linalg.norm(src[s - 1] - det[d - 1]) > MAX_SEPARATION:
            continue
        for w in (1, 2):                                                    # both wavelengths side by side
            columns.append(np.flatnonzero((sources == s) & (detectors == d) & (wl_index == w))[0])
            names.append(f"S{s}_D{d} {wavelengths[w - 1]:g}")
    info = mne.create_info(names, 1.0 / np.median(np.diff(t)), "fnirs_cw_amplitude")
    for ch in info["chs"]:
        pair, wl = ch["ch_name"].split(" ")
        s, d = (int(n[1:]) for n in pair.split("_"))
        ch["loc"][3:6], ch["loc"][6:9] = src[s - 1], det[d - 1]
        ch["loc"][:3] = (src[s - 1] + det[d - 1]) / 2
        ch["loc"][9] = float(wl)
    return mne.io.RawArray(x[:, columns].T, info, verbose=False)


def load_language_production_data(zip_path=ZIP, verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in the zip."""
    outputs = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith("_nirs.snirf")):
            raw = read_run(z.read(name))
            sfreq = raw.info["sfreq"]
            raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
            if raw is None:
                continue
            record = raw_to_record(raw)
            stem = Path(name).stem
            record["subject_id"] = stem.split("_")[0]                       # e.g. sub-01
            outputs[stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_language_production_data()
