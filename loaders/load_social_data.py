"""
Chen, Y., Niu, H., Ao, N., Xu, J., Ming, Q., Teng, X., Yang, B., Lian, Z., Jiang, X., Yang, Y. (2025). An fNIRS
hyperscanning dataset on the modulation of synchrony by social relationship during social touch. Scientific Data
12, 1975. doi:10.1038/s41597-025-06233-9. Data: OSF doi:10.17605/OSF.IO/SM7YT.

47 female dyads (24 friend and 23 stranger pairs) viewing negative and neutral pictures, first without touch and
then holding hands. For each participant, three NIRStar recordings: block1 (resting state and no-touch session,
28-40 min), block2 (touch session, 25-35 min) and block3 (final resting state, ~6 min): 282 recordings.
NIRScout, 785/830 nm, 7.81 Hz: 8 sources and 7 detectors forming 20 channels (30 mm) over the prefrontal
cortex, NIRx's standard "Prefrontal 8x8" montage (the same probeInfo.mat as Parents).
Each optode gets the NIRx ICBM152 MNI position of its 10-20 position (authors' preprocessing.py; S5 on Fpz),
and Beer–Lambert uses the 30-mm separation of the header, as for Parents.
"""

from pathlib import Path

import mne
import numpy as np

from .utils import DATA_ROOT, print_summary, raw_to_record

SD_DISTANCE = 0.030  # m, every channel (ChanDis in the .hdr)

# optode: (10-20 position, from the authors' preprocessing.py; NIRx ICBM152 MNI coordinates in mm)
OPTODES = {
    "S1": ("F3", (-51.830, 55.753, 37.164)),
    "S2": ("AF7", (-54.498, 68.788, -18.293)),
    "S3": ("AF3", (-35.609, 78.819, 16.092)),
    "S4": ("Fz", (-0.235, 62.434, 65.513)),
    "S5": ("Fpz", (0.181, 89.249, -7.826)),
    "S6": ("AF4", (35.272, 78.970, 16.408)),
    "S7": ("F4", (51.167, 56.349, 37.498)),
    "S8": ("AF8", (54.021, 69.084, -17.587)),
    "D1": ("F5", (-64.987, 49.590, 10.244)),
    "D2": ("F1", (-29.480, 60.438, 57.351)),
    "D3": ("Fp1", (-29.230, 85.182, -14.189)),
    "D4": ("AFz", (-0.040, 83.390, 31.683)),
    "D5": ("F2", (28.847, 60.759, 57.469)),
    "D6": ("Fp2", (29.184, 85.158, -13.797)),
    "D7": ("F6", (64.380, 50.370, 10.803)),
}
OPTODE_XYZ = {name: np.array(xyz) / 1000.0 for name, (_, xyz) in OPTODES.items()}  # mm -> m


def load_social_data(main_dir=DATA_ROOT / "social", verbose=True):
    """Return ``{recording_name: record}`` for every NIRStar folder (dyad/block/subject) in ``main_dir``."""
    outputs = {}
    for folder in sorted(Path(main_dir).glob("*/block*/Subject*")):
        dyad, block, subject = folder.parts[-3:]
        hdr = next(folder.glob("*.hdr"))  # NIRS.hdr (dyad 1008, block1 uses another file name)
        raw = mne.io.read_raw_nirx(hdr, preload=True, verbose="error")  # "error": the header date is unreadable
        record = raw_to_record(raw, optode_xyz=OPTODE_XYZ, fix_file_positions=True, sd_distance=SD_DISTANCE)
        record["subject_id"] = f"{dyad}_{subject}"  # e.g. 1003_Subject1
        outputs[f"{dyad}_{block}_{subject}"] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_social_data()
