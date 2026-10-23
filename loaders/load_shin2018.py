"""
Shin, J., von Lühmann, A., Kim, D.-W., Mehnert, J., Hwang, H.-J., Müller, K.-R. (2018). Simultaneous
acquisition of EEG and NIRS during cognitive tasks for an open access dataset. Scientific Data 5, 180003.
doi:10.1038/sdata.2018.3. Raw NIRx files from https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/

26 participants x 3 tasks (n-back, discrimination/selection response, word generation) x 3 sessions of
8-12 min, one NIRStar folder per session: 234 recordings. NIRScout, 760/850 nm, 10.4 Hz: 16 sources and
16 detectors forming 36 channels (30 mm) over the prefrontal, motor, parietal and occipital cortex.
MNE cannot read these folders (no probeInfo.mat), so the raw intensities are read with ``utils.read_nirstar``.
No 3D positions were digitized. Each channel is named by the 10-5 position it is centred on (authors' files
and paper) and gets the MNI position of that label on the NIRx ICBM152 template, as in Baby and Parents.
AF5h and AF6h are not in the NIRx table: they are the midpoints of AF5-AF3 and AF4-AF6, as 10-5 "h"
positions are defined.
"""

from pathlib import Path

import numpy as np

from .utils import DATA_ROOT, print_summary, raw_to_record, read_nirstar

SD_DISTANCE = 0.030  # m, every channel (paper; ChanDis in the .hdr)

# Task of the 9 session folders of a participant, in folder order (dataset description, Table 5)
TASKS = ["nback"] * 3 + ["wg"] * 3 + ["dsr"] * 3
TASKS_VP001_VP002 = ["dsr"] * 3 + ["wg"] * 3 + ["nback"] * 3

# channel: (10-5 label, from the authors' conversion script and files; NIRx ICBM152 MNI position in mm)
CHANNELS = {
    # Prefrontal
    "S1_D1": ("AF7", (-54.498, 68.788, -18.293)), "S1_D2": ("AFF5", (-56.347, 62.867, 5.138)),
    "S2_D1": ("AFp7", (-42.615, 78.199, -16.543)), "S2_D2": ("AF5h", (-40.472, 76.343, 7.606)),
    "S2_D3": ("AFp3", (-24.787, 84.788, 3.329)), "S3_D2": ("AFF3h", (-35.639, 71.263, 36.361)),
    "S3_D3": ("AF1", (-19.609, 82.143, 27.460)), "S3_D4": ("AFFz", (-0.144, 74.974, 49.875)),
    "S4_D3": ("AFpz", (0.069, 87.717, 12.215)), "S4_D4": ("AF2", (19.332, 82.196, 27.530)),
    "S4_D9": ("AFp4", (24.721, 84.805, 3.536)), "S9_D4": ("AFF4h", (35.093, 71.531, 36.589)),
    "S9_D9": ("AF6h", (40.118, 76.534, 8.030)), "S9_D10": ("AFF6", (55.843, 63.342, 5.687)),
    "S10_D9": ("AFp8", (42.331, 78.297, -15.976)), "S10_D10": ("AF8", (54.021, 69.084, -17.587)),
    # Motor (around C3 and C4)
    "S5_D5": ("FCC3", (-67.280, 6.436, 58.396)), "S5_D6": ("C3h", (-56.639, -10.426, 78.110)),
    "S6_D5": ("C5h", (-79.957, -12.630, 44.864)), "S6_D6": ("CCP3", (-72.036, -29.818, 65.923)),
    "S11_D11": ("FCC4", (66.383, 7.825, 58.862)), "S11_D12": ("C6h", (79.342, -10.717, 45.557)),
    "S12_D11": ("C4h", (55.612, -9.079, 78.585)), "S12_D12": ("CCP4", (71.538, -28.329, 66.603)),
    # Parietal (around P3 and P4)
    "S7_D7": ("CPP3", (-64.664, -65.552, 62.557)), "S7_D8": ("P3h", (-46.705, -82.308, 68.435)),
    "S8_D7": ("P5h", (-65.910, -80.024, 41.609)), "S8_D8": ("PPO3", (-49.317, -94.481, 47.424)),
    "S13_D13": ("CPP4", (64.597, -64.616, 63.414)), "S13_D14": ("P6h", (66.270, -79.136, 42.386)),
    "S14_D13": ("P4h", (46.619, -81.692, 69.104)), "S14_D14": ("PPO4", (49.453, -93.982, 48.206)),
    # Occipital (around POz)
    "S15_D15": ("PPOz", (-0.318, -96.600, 69.353)), "S15_D16": ("PO1", (-20.890, -106.296, 47.981)),
    "S16_D15": ("PO2", (20.720, -106.154, 48.397)), "S16_D16": ("POOz", (-0.127, -114.608, 33.665)),
}
CHANNEL_XYZ = {name: np.array(xyz) / 1000.0 for name, (_, xyz) in CHANNELS.items()}  # mm -> m


def read_session(folder):
    """Raw intensity of one NIRStar session folder as an MNE Raw, each channel at its MNI position."""
    raw = read_nirstar(folder)                                     # 36 channels x 2 wavelengths, 10.4167 Hz
    for ch in raw.info["chs"]:
        xyz = CHANNEL_XYZ[ch["ch_name"].split(" ")[0]]
        ch["loc"][:3] = xyz                                        # channel position (MNI)
        ch["loc"][3:6] = xyz - [SD_DISTANCE / 2, 0, 0]             # source and detector 30 mm apart around it:
        ch["loc"][6:9] = xyz + [SD_DISTANCE / 2, 0, 0]             # only their distance is used (Beer–Lambert)
    return raw


def load_shin2018_data(main_dir=DATA_ROOT / "shin2018", verbose=True):
    """Return ``{recording_name: record}`` for every session folder in ``main_dir``."""
    outputs = {}
    for sub_dir in sorted(Path(main_dir).glob("VP0??")):                         # VP001 ... VP026 (not the .zip files)
        tasks = TASKS_VP001_VP002 if sub_dir.name in ("VP001", "VP002") else TASKS
        for k, folder in enumerate(sorted(p for p in sub_dir.iterdir() if p.is_dir())):
            record = raw_to_record(read_session(folder))
            record["subject_id"] = sub_dir.name                                  # VP0XX
            outputs[f"{sub_dir.name}_{tasks[k]}_{k % 3 + 1}"] = record           # e.g. VP001_dsr_1
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_shin2018_data()
