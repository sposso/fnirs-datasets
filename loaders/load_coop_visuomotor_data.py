"""
Zheng, Y., Tian, K., Shan, B., Han, T., Tian, B., Liu, Q., Shen, X., Peng, C., Su, P., Wang, D. (2026). A dyadic fNIRS
hyperscanning dataset for cooperative visuomotor control with haptic interaction. ScienceDB,
doi:10.57760/sciencedb.46254 (v1.0.0; the accompanying data descriptor was not yet available).

64 healthy young adults in 32 same-sex dyads colored a picture together with two haptic devices, in four runs of
10-13 min (two collaboration modes x two difficulty levels). NirScan (730/808/850 nm, 11 Hz); one recording per dyad
and run holds both members: channels 1-26 (S1-S12, D1-D11) belong to member A and channels 27-52 (S13-S24, D12-D22)
to member B, over the left and right prefrontal cortex, the left sensorimotor cortex and the right temporoparietal
junction, 30 mm apart (the files' 2D layout). Each file is split into one recording per member: 256 recordings.
As for HEFMI-ICH (same device), 730 and 850 nm are used and 808 nm is dropped (MNE's Beer–Lambert takes two
wavelengths; MNE also cannot read 3-wavelength SNIRF files, so the 730/850-nm intensities are read with h5py).
No author flags are shipped; all channels are kept.
Positions: the SNIRF files hold only a 2D layout. task-coopvm_channel-pairs.tsv gives the MNI coordinates (AtlasViewer)
of the 26 channel positions with the source-detector pair of each member, the same for every participant; used as
given (channel_pos), with a 30-mm source-detector distance.
"""

import csv
from pathlib import Path

import h5py
import mne
import numpy as np

from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

SD_DISTANCE = 0.030       # m (files' 2D layout)
WAVELENGTHS = (730, 850)  # nm kept; 808 nm dropped


def channel_tables(main_dir):
    """Per member, ``{"S1_D1": MNI xyz in metres}`` from task-coopvm_channel-pairs.tsv."""
    rows = list(csv.DictReader(open(Path(main_dir) / "task-coopvm_channel-pairs.tsv"), delimiter="\t"))
    tables = {"A": {}, "B": {}}
    for r in rows:
        xyz = np.array([float(r[k]) for k in ("mni_x", "mni_y", "mni_z")]) / 1000.0
        for member in ("A", "B"):
            tables[member][r[f"participant_{member.lower()}_source_detector"].replace("-", "_")] = xyz
    return tables


def read_member(path, table):
    """Raw intensity (730/850 nm) of one member's channels in a dyad SNIRF file as an MNE Raw (no positions yet)."""
    with h5py.File(path, "r") as f:
        data = f["nirs/data1/dataTimeSeries"][()]
        time = f["nirs/data1/time"][()]
        wavelengths = f["nirs/probe/wavelengths"][()]
        names, columns = [], []
        for i in range(data.shape[1]):
            m = f[f"nirs/data1/measurementList{i + 1}"]
            s, d, w = (int(np.ravel(m[k][()])[0]) for k in ("sourceIndex", "detectorIndex", "wavelengthIndex"))
            if f"S{s}_D{d}" in table and int(wavelengths[w - 1]) in WAVELENGTHS:
                names.append(f"S{s}_D{d} {int(wavelengths[w - 1])}")
                columns.append(i)
    order = sorted(range(len(names)), key=lambda k: (names[k].split(" ")[0], names[k]))  # wavelengths of a pair together
    info = mne.create_info([names[k] for k in order], 1 / np.median(np.diff(time)), "fnirs_cw_amplitude")
    for ch in info["chs"]:
        ch["loc"][9] = float(ch["ch_name"].split(" ")[1])                  # wavelength (nm)
        pair = ch["ch_name"].split(" ")[0]
        ch["loc"][:3] = table[pair]                                         # channel midpoint (MNI); the source and
        ch["loc"][3:6] = table[pair] - [SD_DISTANCE / 2, 0, 0]             # detector 30 mm apart around it: only
        ch["loc"][6:9] = table[pair] + [SD_DISTANCE / 2, 0, 0]             # their distance is used (Beer–Lambert)
    return mne.io.RawArray(data[:, [columns[k] for k in order]].T, info, verbose=False)


def load_coop_visuomotor_data(main_dir=DATA_ROOT / "coop_visuomotor", verbose=True):
    """Return ``{recording_name: record}``: one record per member and run."""
    tables = channel_tables(main_dir)
    outputs = {}
    for path in sorted(Path(main_dir).glob("dyad-*/nirs/*.snirf")):
        dyad, run = path.name.split("_")[0], path.name.split("_")[2]       # e.g. dyad-001, run-01
        for member, table in tables.items():
            part = read_member(path, table)
            sfreq = part.info["sfreq"]
            part = fill_or_drop(part, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
            if part is None:
                continue
            record = raw_to_record(part, channel_pos=table)
            record["subject_id"] = f"{dyad}{member}"                         # e.g. dyad-001A
            outputs[f"{dyad}{member}_{run}"] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_coop_visuomotor_data()
