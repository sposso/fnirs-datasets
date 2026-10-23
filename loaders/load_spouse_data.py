"""
Azhari, A., Lim, M., Bizzego, A., Gabrieli, G., Bornstein, M. H., Esposito, G. (2020). Physical presence of spouse
enhances brain-to-brain synchrony in co-parenting couples. Scientific Reports 10, 7569.
doi:10.1038/s41598-020-63596-2. Data: DR-NTU doi:10.21979/N9/KF1JOG (CC BY-NC 4.0).

24 mother-father couples listening to infant and adult vocalisations and static noise (6 sounds x 3 trials of 15 s),
once together in the same room and once separately. Four NIRStar recordings per couple (together: SSCxxC/SSCxxF and
SSCxxC/SSCxxM; separate: SSCxxF and SSCxxM; F = father, M = mother), 10-29 min each: 96 recordings.
NIRSport, 760/850 nm, 7.81 Hz: 8 sources and 7 detectors forming 20 channels (30 mm) over the prefrontal cortex,
NIRx's standard "Prefrontal 8x8" montage (the same probeInfo.mat as Parents and Social).
Read with ``utils.read_nirstar``, as MNE cannot read the two NIRStar 14.2 recordings. Samples flagged as saturated
(NaN, in 42 recordings) are handled as for Parents (utils.fill_or_drop): a missing start or end of up to 1 min is
cut off, gaps of up to 1 s are filled linearly and channels with a longer gap are dropped; every recording is kept
with the channels it has left.
Each optode gets the NIRx ICBM152 MNI position of its 10-20 position, and Beer–Lambert uses the 30-mm separation of
the header, as for Parents and Social.
"""

from pathlib import Path

import numpy as np

from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record, read_nirstar

SD_DISTANCE = 0.030  # m, every channel (ChanDis in the .hdr)
MAX_GAP = 8          # longest gap that is filled, in samples (~1 s at 7.8 Hz)
MAX_EDGE = 469       # longest missing start or end that is cut off, in samples (1 min)

# optode: (10-20 position of NIRx's Prefrontal 8x8 montage, NIRx ICBM152 MNI coordinates in mm)
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


def load_spouse_data(main_dir=DATA_ROOT / "spouse", verbose=True):
    """Return ``{recording_name: record}`` for the 4 recordings of every couple folder in ``main_dir``."""
    outputs = {}
    for couple in sorted(Path(main_dir).glob("SSC??")):                        # SSC12 ... SSC39 (not the .zip files)
        for condition, parent_dirs in (("together", couple / f"{couple.name}C"), ("separate", couple)):
            for folder in sorted(parent_dirs.glob(f"{couple.name}[FM]")):       # SSCxxF (father), SSCxxM (mother)
                raw = fill_or_drop(read_nirstar(folder), MAX_GAP, MAX_EDGE)
                if raw is None:                                                 # no channel left
                    continue
                record = raw_to_record(raw, optode_xyz=OPTODE_XYZ, fix_file_positions=True, sd_distance=SD_DISTANCE)
                record["subject_id"] = folder.name                              # e.g. SSC12F
                outputs[f"{folder.name}_{condition}"] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_spouse_data()
