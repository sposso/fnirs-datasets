"""
Hamann, A., Carstengerdes, N. (2022). Investigating mental workload-induced changes in cortical oxygenation and frontal
theta activity during simulated flights. Scientific Reports 12, 6449. doi:10.1038/s41598-022-10044-y.
Data: PsychArchives doi:10.23668/psycharchives.5291 (fnirs_data_stims.mat).

35 students flew an Airbus A321 simulator while doing an auditory n-back task (0- to 3-back) and an altitude-monitoring
task: 8 blocks of ~3 min with 2-min rests, one recording of 43-51 min each (28.5 h). NIRSport2 (NIRx; 760/850 nm,
10.17 Hz), 8 sources and 7 detectors forming 17 channels of 26-39 mm over the bilateral prefrontal cortex, plus
8 short-separation channels (~8 mm, one per source); all kept.
The file holds a NIRS Brain AnalyzIR toolbox object (nirs.core.Data) per participant with the raw intensity
(read with mat-io; scipy cannot decode MATLAB class objects).
Positions: the paper's Fig. 4 gives the 10-5 label of every source and detector. The stored probe positions match
the NIRx ICBM152 points of those labels after a rigid transform with 0.0 mm residual, so the same transform moves all
optodes (including the short-separation detectors) to MNI.
"""

import json
from pathlib import Path

import matio
import mne
import numpy as np
from mne.transforms import apply_trans

from .utils import DATA_ROOT, NIRX_ICBM152, fill_or_drop, print_summary, raw_to_record, set_optode_positions

SOURCES = ["AFp2", "AFF6h", "FFC4h", "FFC1h", "AF7", "AFF3h", "FFC5h", "FCC3h"]   # S1-S8 (paper, Fig. 4)
DETECTORS = ["AFp1", "AFF5h", "FFC3h", "FFC2h", "AF8", "AFF4h", "FFC6h"]           # D1-D7 (paper, Fig. 4)


def probe_to_mni(optodes):
    """Rigid transform taking the stored probe positions (mm) of S1-S8 and D1-D7 onto the NIRx ICBM152 points of
    their 10-5 labels; returns the transform and the largest residual (mm)."""
    table = json.loads(NIRX_ICBM152.read_text())
    stored = optodes[["X", "Y", "Z"]].to_numpy(float)[:15]                 # S1-S8, D1-D7 (D8-D15 are short)
    target = np.array([table[label] for label in SOURCES + DETECTORS])
    trans = mne.coreg.fit_matched_points(stored, target, scale=0, out="trans")
    return trans, np.linalg.norm(apply_trans(trans, stored) - target, axis=1).max()


def read_participant(data):
    """MNE Raw (raw intensity, optodes in MNI) from one nirs.core.Data object."""
    p = data.properties
    probe = p["probe"].properties
    optodes, link = probe["optodes"], probe["link"]
    trans, residual = probe_to_mni(optodes)
    if residual > 1.0:
        raise ValueError(f"probe does not match the Fig. 4 layout (residual {residual:.1f} mm)")
    names = [str(np.ravel(n)[0]).strip("[]'") for n in optodes["Name"]]   # e.g. Source-0001, Detector-0008
    xyz = apply_trans(trans, optodes[["X", "Y", "Z"]].to_numpy(float)) / 1000.0  # MNI, m
    positions = {("S" if n.startswith("Source") else "D") + str(int(n.split("-")[1])): x for n, x in zip(names, xyz)}
    ch_names = [f"S{int(s)}_D{int(d)} {int(w)}" for s, d, w in link[["source", "detector", "type"]].to_numpy(float)]
    time = np.ravel(p["time"])
    info = mne.create_info(ch_names, 1 / np.median(np.diff(time)), "fnirs_cw_amplitude")
    for ch in info["chs"]:
        ch["loc"][9] = float(ch["ch_name"].split(" ")[1])                  # wavelength (nm)
    set_optode_positions(info, positions)
    return mne.io.RawArray(np.asarray(p["data"], float).T, info, verbose=False)


def load_flight_data(main_dir=DATA_ROOT / "flight_workload", verbose=True):
    """Return ``{recording_name: record}`` for the 35 participants in ``fnirs_data_stims.mat``."""
    participants = matio.load_from_mat(Path(main_dir) / "fnirs_data_stims.mat")["fnirs_data_stims"].ravel()
    outputs = {}
    for data in participants:
        subject = Path(str(np.ravel(data.properties["description"])[0]).replace("\\", "/")).stem  # e.g. MWL21
        raw = read_participant(data)
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        if raw is None:
            continue
        record = raw_to_record(raw)
        record["subject_id"] = subject
        outputs[subject] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_flight_data()
