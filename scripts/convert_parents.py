"""Convert the parent-child hyperscanning recordings (Bizzego et al. 2022, DR-NTU) from NIRStar files to SNIRF.

Each recording folder (<dyad>/child or <dyad>/parent) holds the raw light intensity at 760 nm (.wl1) and
850 nm (.wl2). MNE cannot read 67 of the 248 folders (non-standard event files, NIRStar 14.2), so they are read
with loaders/utils.read_nirstar; the events are not needed.

Missing samples (NaN, in 76 recordings), with loaders/utils.fill_or_drop: the start and end are cut back to where
every channel has data, by at most 1 min (channels that would need more are dropped); inside, gaps of up to 1 s are
filled by linear interpolation (as nirs-toolbox's FixNaNs does) and channels with a longer gap are dropped;
every recording is kept with the channels it has left.

Optode positions: NIRx ICBM152 (MNI) coordinates of the 10-20 positions of NIRx's standard "Prefrontal 8x8"
montage, which every recording uses (probeInfo.mat). When every channel of an optode is dropped, the SNIRF
writer renumbers the later optodes (e.g. D7 is read back as D6); positions and optode labels stay correct.

Run once from the repository root:  python -m scripts.convert_parents
Reads $FNIRS_DATA_ROOT/parents_download/ and writes $FNIRS_DATA_ROOT/parents/<session>/<dyad>_<role>_<session>.snirf;
the original files are not touched.
"""

from datetime import datetime, timezone

import numpy as np
from mne_nirs.io import write_raw_snirf

from loaders.utils import DATA_ROOT, fill_or_drop, read_nirstar

SRC = DATA_ROOT / "parents_download"  # original NIRStar folders
DST = DATA_ROOT / "parents"           # converted SNIRF files
SESSIONS = {"mothers": "video", "fathers": "video", "mothersplay/mothers": "play", "fathersplay/fathers": "play"}
MAX_GAP = 8        # longest gap that is filled, in samples (~1 s at 7.8 Hz)
MAX_EDGE = 469     # longest missing start or end that is cut off, in samples (1 min at 7.8 Hz)

# optode: (10-20 position in NIRx's Prefrontal 8x8 montage, NIRx ICBM152 MNI coordinates in mm)
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


def write_snirf(raw, subject, out):
    """Write raw intensity to a SNIRF file, each optode at its template position."""
    for ch in raw.info["chs"]:
        source, detector = ch["ch_name"].split(" ")[0].split("_")   # "S1_D1 760" -> "S1", "D1"
        ch["loc"][3:6] = OPTODE_XYZ[source]                          # source position (m)
        ch["loc"][6:9] = OPTODE_XYZ[detector]                        # detector position (m)
        ch["loc"][:3] = (ch["loc"][3:6] + ch["loc"][6:9]) / 2        # channel midpoint
    raw.set_meas_date(datetime(2000, 1, 1, tzinfo=timezone.utc))    # placeholder date, required by write_raw_snirf
    raw.info["subject_info"] = {"his_id": subject}                   # subject name, required by write_raw_snirf
    out.parent.mkdir(parents=True, exist_ok=True)
    write_raw_snirf(raw, out)


if __name__ == "__main__":
    for folder, session in SESSIONS.items():                         # mother/father dyads, video/play sessions
        for rec in sorted((SRC / folder).glob("*/*")):               # <dyad>/child and <dyad>/parent
            dyad, role = rec.parent.name, rec.name                   # e.g. "NSFD1", "child"
            raw = fill_or_drop(read_nirstar(rec), MAX_GAP, MAX_EDGE)
            if raw is None:                                          # no channel left
                print(f"skipped {dyad}_{role}_{session}: no channel left")
                continue
            write_snirf(raw, f"{dyad}_{role}", DST / session / f"{dyad}_{role}_{session}.snirf")
