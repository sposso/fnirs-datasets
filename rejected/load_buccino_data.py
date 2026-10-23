"""
Buccino, A. P., Keles, H. O., Omurtag, A. (2016). Hybrid EEG-fNIRS asynchronous brain-computer interface for
multiple motor tasks. PLoS ONE 11(1), e0146610.
Data: figshare doi:10.6084/m9.figshare.1619640 (S1-S8) and doi:10.6084/m9.figshare.1619641 (S9-S15), read from the
two zip files without extracting them.

15 right-handed men raising the right or left arm or gripping with the right or left hand (5 blocks of 20 trials),
with simultaneous EEG: one recording of 24-27 min each. NIRScout 8-16 (NIRx; 760/850 nm), 10.42 Hz, 12 sources and
12 detectors, 34 channels over the motor cortex; all kept. Raw light intensity (nirs wl1, wl2) is used. The paper
gives only a maximum source-detector distance of 3.4 cm: the nominal 30 mm is assumed for Beer-Lambert.
Rejected for the main set: no positions. Each channel is named only by its offset from a 10-20 electrode (e.g. Fc3A,
C3L); these labels are kept as provided in "channel_labels"; channel_xyz is None.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

from loaders.utils import DATA_ROOT, fill_or_drop, intensity_to_raw, print_summary, raw_to_record

SD_DISTANCE = 0.03       # metres (assumed; the paper gives a maximum of 3.4 cm)


def load_buccino_data(main_dir=DATA_ROOT / "buccino", verbose=True):
    """Return ``{recording_name: record}`` for every participant file in the zip files of ``main_dir``."""
    outputs = {}
    for zip_path in sorted(Path(main_dir).glob("*.zip")):
        with zipfile.ZipFile(zip_path) as z:
            for member in sorted(n for n in z.namelist() if n.endswith(".mat")):
                nirs = sio.loadmat(io.BytesIO(z.read(member)), squeeze_me=True, struct_as_record=False)["data"][1]
                sfreq, labels, wl1, wl2 = float(nirs[0]), list(nirs[2]), nirs[4], nirs[5]   # (time x channels)
                raw = intensity_to_raw(np.stack([wl1.T, wl2.T]), sfreq, (760, 850), SD_DISTANCE)
                raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min edges
                record = raw_to_record(raw, allow_2d=True)
                kept = [int(ch.split("_")[0][1:]) - 1 for ch in raw.ch_names[::2]]  # channels left by fill_or_drop
                record["channel_xyz"] = None
                record["channel_labels"] = [labels[i] for i in kept]
                record["subject_id"] = Path(member).stem                            # S1 ... S15
                record["coord_frame"] = "none; channel labels = offsets from 10-20 electrodes, as provided"
                outputs[record["subject_id"]] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_buccino_data()
