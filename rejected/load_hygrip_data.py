"""
Ortega, P., Zhao, T., Faisal, A. A. (2020). HYGRIP: Full-stack characterization of neurobehavioral signals (fNIRS,
EEG, EMG, force, and breathing) during a bimanual grip force control task. Front. Neurosci. 14, 919.
Data: figshare doi:10.6084/m9.figshare.12383639 (hygrip.h5, MIT).

14 adults (A-N) performing a bimanual grip force control task: one continuous recording of 19-26 min each. NIRScout
(NIRx; 760/850 nm), 12.5 Hz, 24 channels at 3 cm, 12 around C3 and 12 around C4; all kept. Raw light intensity
(oi1, oi2) is used; the authors' HbO/HbR (oxy, dxy) are not.
Rejected for the main set: no 3D positions. The file gives each channel only as a polar offset (angle in degrees,
radius in m) from C3 (Ch1-12) or C4 (Ch13-24), with an undocumented angle convention (attribute scalp_chan_locs).
These values are kept as provided in "channel_polar"; channel_xyz is None.
"""

from pathlib import Path

import h5py
import numpy as np

from loaders.utils import DATA_ROOT, fill_or_drop, intensity_to_raw, print_summary, raw_to_record

SD_DISTANCE = 0.03       # metres (nirs_SD: d[cm]=3)


def load_hygrip_data(main_dir=DATA_ROOT / "hygrip", verbose=True):
    """Return ``{recording_name: record}`` for every participant of ``hygrip.h5`` in ``main_dir``."""
    outputs = {}
    with h5py.File(Path(main_dir) / "hygrip.h5", "r") as f:
        polar = np.array([line.split("\t")[1:3] for line in f.attrs["scalp_chan_locs"].strip().splitlines()], float)
        for name in sorted(f.keys()):
            intensity = np.stack([f[name]["oi1"][()], f[name]["oi2"][()]])        # 760 nm, 850 nm
            raw = intensity_to_raw(intensity, float(f.attrs["oi1_sfreq"]), (760, 850), SD_DISTANCE)
            sfreq = raw.info["sfreq"]
            raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
            record = raw_to_record(raw, allow_2d=True)
            kept = [int(ch.split("_")[0][1:]) - 1 for ch in raw.ch_names[::2]]     # channels left by fill_or_drop
            record["channel_xyz"] = None
            record["channel_polar"] = polar[kept]                               # angle (deg), radius (m)
            record["subject_id"] = name                                           # A ... N
            record["coord_frame"] = "polar offsets (angle deg, radius m) from C3 (Ch1-12) / C4 (Ch13-24), as provided"
            outputs[name] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_hygrip_data()
