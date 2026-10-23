"""
Zhang, X., Wang, Q., Li, J., Gao, X., Li, B., Nie, B., Wang, J., Zhou, Z., Yang, Y., Wang, H. (2024).
An fNIRS dataset for driving risk cognition of passengers in highly automated driving scenarios.
Scientific Data 11, 546. doi:10.1038/s41597-024-03353-6. OpenNeuro ds004973.

20 participants x 12 tasks (~13-min simulated automated drives watched as a passenger): 222 recordings.
Artinis OctaMon over the prefrontal cortex: 8 sources and 2 detectors forming 8 channels (35 mm),
760/850 nm, 50 Hz.
The files hold only a flat 2D layout whose distances are 10x too small (3.5 instead of 35 mm), so
Beer–Lambert uses the 35-mm separation of every channel, and each channel gets the MNI coordinates of
the paper's Table 3.
25 recordings were uploaded truncated; their signals were recovered with scripts/recover_driving.py
and are read from ``recovered_dir`` instead.
"""

from pathlib import Path

import numpy as np

from .utils import DATA_ROOT, print_summary, snirf_to_record

# Channel MNI coordinates in mm (paper, Table 3: OxySoft with Artinis' predefined digitization)
CHANNEL_MNI = {
    "S1_D1": (57.00, 50.56, 19.33), "S2_D1": (57.12, 56.10, 1.26),
    "S3_D1": (38.41, 70.37, 19.56), "S4_D1": (32.83, 76.89, -0.91),
    "S5_D2": (-31.45, 75.44, 21.04), "S6_D2": (-33.04, 77.66, 0.59),
    "S7_D2": (-56.11, 54.37, 19.49), "S8_D2": (-55.73, 60.39, -1.00),
}
CHANNEL_POS = {name: np.array(xyz) / 1000.0 for name, xyz in CHANNEL_MNI.items()}  # mm -> m
SD_DISTANCE = 0.035  # m, every OctaMon channel


def load_driving_data(main_dir=DATA_ROOT / "driving", recovered_dir=DATA_ROOT / "driving_recovered", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*.snirf")):
        recovered = Path(recovered_dir) / path.relative_to(main_dir)
        if recovered.exists():  # truncated on OpenNeuro: use the recovered signals
            path = recovered
        record = snirf_to_record(path, sd_distance=SD_DISTANCE, channel_pos=CHANNEL_POS)
        record["subject_id"] = path.parts[-3]  # sub-X
        outputs[path.stem] = record

    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_driving_data()
