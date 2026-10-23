"""
Zhang, H., Hu, Y., Li, Y., Zhang, S., Li, X., Zhao, C. (2025). Simultaneous Dataset of Brain, Eye and Hand during
Visuomotor Tasks. Scientific Data 12, 189. doi:10.1038/s41597-024-04227-7. Data: OSF doi:10.17605/osf.io/cfdsz
(NIR/RAW.zip, extracted to raw_csv/).

50 adults found a target aircraft among distractors and pointed at it with a joystick, in 6 blocks (each with a 2-min
rest, an easy and a hard task), in one session of 1.6-2.4 h. Hitachi ETG-4000, 695/830 nm, 10 Hz: two 3x5 probe
sets, frontal (Probe1, channels 1-22) and parietal (Probe2, channels 23-44), recorded simultaneously but saved to two
CSV files (sub<N>_P1.csv, sub<N>_P2.csv, identical time stamps), joined here into one 44-channel recording.
Positions: the MNI coordinates of the 44 channels in the paper's Table 2 (co-registered with MRI, on the cortex),
used as given; source-detector distance 30 mm (paper).
Bad channels: the channels marked in each file's "Exception Ch" header line (set on the device during setup; they
show no heartbeat signal) are dropped. A start or end where a channel is zero or negative is cut off (up to 1 min)
instead of dropping the channel; gaps of up to 1 s are filled linearly and channels with longer gaps are dropped; every
recording is kept with the channels it has left.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

SFREQ = 10.0               # Hz
WAVELENGTHS = (695, 830)   # nm (paper; the files also list each channel's measured values, ~700 and ~832 nm)
SD_DISTANCE = 0.030        # m, every channel (paper)
MAX_GAP = 10               # longest gap that is filled, in samples (1 s)
MAX_EDGE = 600             # longest missing start or end that is cut off, in samples (1 min)

# Channel MNI coordinates in mm (paper, Table 2): channels 1-22 frontal (Probe1), 23-44 parietal (Probe2)
CHANNEL_MNI = [
    (60, 2, 35), (28, 7, 67), (-29, 7, 67), (-59, 3, 35), (61, 13, 18), (39, 19, 54), (0, 22, 69), (-38, 19, 54),
    (-60, 13, 18), (53, 28, 27), (26, 34, 56), (-24, 34, 55), (-51, 28, 28), (53, 36, 11), (33, 44, 40), (0, 47, 53),
    (-32, 44, 41), (-51, 37, 13), (42, 49, 17), (21, 55, 37), (-19, 56, 37), (-40, 50, 17),
    (-56, -51, 41), (-27, -51, 70), (27, -51, 70), (55, -51, 40), (-55, -62, 26), (-35, -64, 57), (-1, -64, 68),
    (34, -63, 57), (53, -63, 26), (-44, -73, 33), (-21, -74, 53), (20, -74, 54), (42, -73, 33), (-43, -80, 14),
    (-24, -82, 38), (-1, -82, 44), (22, -82, 38), (41, -80, 14), (-30, -87, 18), (-12, -88, 30), (10, -89, 29),
    (28, -87, 18),
]
CHANNEL_XYZ = np.array(CHANNEL_MNI) / 1000.0  # mm -> m


def read_probe(path):
    """Raw intensity (time x 44 columns: 22 channels x 2 wavelengths) and the 'Exception Ch' flags of one ETG-4000 CSV."""
    lines = path.read_text(encoding="latin-1").splitlines()
    start = lines.index("Data") + 2                                        # first row of numbers
    exception = next(line for line in lines if line.startswith("Exception Ch")).split(",")[1:23]
    data = pd.read_csv(path, skiprows=start, header=None, usecols=range(1, 45), encoding="latin-1").to_numpy(float)
    return data, np.array(exception, int) == 1


def make_raw(data, keep):
    """MNE Raw of the kept channels, each at its MNI position with source and detector 30 mm apart around it."""
    channels = np.flatnonzero(keep)
    names = [f"S{c + 1}_D{c + 1} {wl}" for c in channels for wl in WAVELENGTHS]  # one source-detector pair per channel
    info = mne.create_info(names, SFREQ, "fnirs_cw_amplitude")
    for ch in info["chs"]:
        pair, wl = ch["ch_name"].split(" ")
        xyz = CHANNEL_XYZ[int(pair.split("_")[0][1:]) - 1]
        ch["loc"][:3] = xyz                                                 # channel position (MNI)
        ch["loc"][3:6] = xyz - [SD_DISTANCE / 2, 0, 0]                      # only the source-detector distance
        ch["loc"][6:9] = xyz + [SD_DISTANCE / 2, 0, 0]                      # matters (Beer–Lambert)
        ch["loc"][9] = float(wl)                                            # wavelength (nm)
    return mne.io.RawArray(data[:, np.repeat(keep, 2)].T, info, verbose=False)


def load_visuomotor_data(main_dir=DATA_ROOT / "visuomotor", verbose=True):
    """Return ``{recording_name: record}``, one 44-channel recording per participant in ``main_dir``."""
    outputs = {}
    for p1 in sorted(Path(main_dir).glob("sub*_P1.csv"), key=lambda p: int(p.name[3:].split("_")[0])):
        subject = p1.name.split("_")[0]                                     # sub<N>
        frontal, bad_frontal = read_probe(p1)
        parietal, bad_parietal = read_probe(p1.with_name(f"{subject}_P2.csv"))
        data = np.hstack([frontal, parietal])                               # time x 88: channels 1-22, then 23-44
        keep = ~np.concatenate([bad_frontal, bad_parietal])                 # drop the exception channels
        raw = fill_or_drop(make_raw(data, keep), MAX_GAP, MAX_EDGE)
        if raw is None:                                                     # no channel left
            continue
        record = raw_to_record(raw)
        record["subject_id"] = subject
        outputs[subject] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_visuomotor_data()
