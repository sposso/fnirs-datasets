"""
Bak, S., Park, J., Shin, J., Jeong, J. (2019). Open-access fNIRS dataset for classification of unilateral finger- and
foot-tapping. Electronics 8(12), 1486.
Data: figshare doi:10.6084/m9.figshare.9783755.v2 (30 BBCI-format files "fNIRS NN.mat").

30 adults performing right-hand finger, left-hand finger and foot tapping (25 trials each), three sessions joined
end to end in one recording of ~37.5 min per participant. LIGHTNIRS (Shimadzu), 8 sources and 8 detectors, 20
channels at 30 mm around C3 and C4, 13.33 Hz; all kept. Only the HbO/HbR of the Shimadzu software are shared (ch1-20
oxy, ch21-40 deoxy, mM*cm): divided by 18 cm (30 mm x partial pathlength factor 6) to get M, as for the other
Shimadzu datasets.
Rejected for the main set: no positions. mnt.x, mnt.y and mnt.pos_3d are NaN in every file, so no positions are
stored (channel_xyz = None).
"""

from pathlib import Path

import numpy as np
import scipy.io as sio

from loaders.utils import DATA_ROOT, PPF, hb_to_record, print_summary

SD_DISTANCE = 0.03                              # metres (paper)
TO_MOLAR = 1e-3 / (SD_DISTANCE * 100 * PPF)    # mM*cm -> M, dividing by 18 cm


def load_tapping_data(main_dir=DATA_ROOT / "tapping", verbose=True):
    """Return ``{recording_name: record}`` for every ``fNIRS NN.mat`` file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("fNIRS *.mat")):
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        signals = np.array([mat[f"ch{i}"] for i in range(1, 41)]) * TO_MOLAR    # 20 oxy, then 20 deoxy
        record = hb_to_record(signals[:20], signals[20:], mat["nfo"].fs, None, np.full(20, SD_DISTANCE))
        record["subject_id"] = f"sub-{path.stem.split()[1]}"                   # sub-01 ... sub-30
        record["coord_frame"] = "none (mnt positions are NaN)"
        outputs[record["subject_id"]] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_tapping_data()
