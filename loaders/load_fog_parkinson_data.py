"""
Cockx, H. M., Oostenveld, R., Flórez R, Y. A., Bloem, B. R., Cameron, I. G. M., van Wezel, R. J. A. (2024). Freezing
of gait in Parkinson's disease is related to imbalanced stopping-related cortical activity. Brain Communications 6(5),
fcae259. doi:10.1093/braincomms/fcae259.
Data: Donders Repository DSC_626810_0022_600 v1, "Cortical activity measured with fNIRS related to freezing of gait in
Parkinson's disease" (doi:10.34973/k7ce-6n58, CC BY-SA 4.0), BIDS folder bids_raw (motion data not downloaded).

22 people with Parkinson's disease and daily freezing of gait and 22 controls walked a corridor with 180-degree turns,
a narrow doorway, and instructed stops and starts; about 4 runs of ~6.5 min each (183 runs, 2.0-15.8 min).
Two Artinis Brite24 devices (LED sources, 760/850 nm, photodiode detectors), recorded at 50 Hz and resampled to 60 Hz
by the authors: 32 long (~30 mm) and 16 short (~10 mm) channels over M1, PMC, SMA, PFC and PPC.
Dropped: channels marked "bad" in each run's channels.tsv (the authors' signal-quality exclusions).
Data: optical density exported through FieldTrip, as in the finger/leg dataset of the same lab (load_finger_leg_data):
base-10, multiplied by ln(10) for MNE's Beer–Lambert.
Positions: digitized for each participant from a 3D head scan, in mm (optodes.tsv, coordsystem.json; BIDS "CTF" frame,
with the nasion, inion, preauricular points and Cz). The SNIRF files label the same values as cm, so the positions read
by MNE are divided by 10. Fitted to fsaverage and returned in MNI by utils.optode_mni.
"""

import csv
from pathlib import Path

import h5py
import mne
import numpy as np

from .utils import DATA_ROOT, optode_mni, print_summary, raw_to_record


def bad_pairs(path):
    """Pairs marked "bad" in the run's channels.tsv, named as MNE names them ("S1_D2"; S/D numbers follow the
    SNIRF sourceLabels/detectorLabels order). A pair is dropped if either wavelength is bad."""
    with h5py.File(path, "r") as f:
        sources = {label.decode(): i + 1 for i, label in enumerate(f["nirs/probe/sourceLabels"][()])}
        detectors = {label.decode(): i + 1 for i, label in enumerate(f["nirs/probe/detectorLabels"][()])}
    with open(str(path).replace("_nirs.snirf", "_channels.tsv")) as f:
        rows = [row for row in csv.DictReader(f, delimiter="\t") if row["status"] == "bad"]
    return {f"S{sources[row['source']]}_D{detectors[row['detector']]}" for row in rows}


def load_fog_parkinson_data(main_dir=DATA_ROOT / "fog_parkinson", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*_nirs.snirf")):
        subject = path.name.split("_")[0]                                   # e.g. sub-PD06
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        for ch in raw.info["chs"]:
            ch["loc"][:9] /= 10.0                                           # positions stored in mm but labeled cm
        bad = bad_pairs(path)
        raw.drop_channels([name for name in raw.ch_names if name.split(" ")[0] in bad])
        optode_xyz = optode_mni(raw, path.parent / f"{subject}_coordsystem.json")
        raw.apply_function(lambda x: x * np.log(10), channel_wise=False)    # base-10 OD -> natural-log OD
        record = raw_to_record(raw, optode_xyz=optode_xyz)
        record["subject_id"] = subject
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_fog_parkinson_data()
