"""
Hakim, U., Noah, J. A., Zhang, X., Gunasekara, N., Hamilton, A., Pinti, P., Tachtsidis, I., Hirsch, J.
(2026). Investigating face processing in online interactions via UK-US hyperscanning using fNIRS.
Imaging Neuroscience 4. doi:10.1162/IMAG.a.1101. Data: Dryad doi:10.5061/dryad.2ngf1vj10
(the extracted hyper_ folder is the dataset folder).

40 participants (20 dyads, one partner at Yale and one at UCL) gazing at their partner's face over Zoom,
live or as a static photo. Shimadzu LABNIRS (780/805/830 nm):
- UCL: one ~30-min recording per participant at 12.35 Hz, 88 channels (76 for S19-S21), 3 cm.
- Yale: ~8 runs of 2 min per participant at 8.13 or 8.55 Hz; 3 or 2.75 cm depending on head size.
The files hold only HbO/HbR computed by the Shimadzu software (unfiltered, starting at 0), as
concentration x path length, assumed to be in mM*cm. We divide by the path length used for the other
datasets (30 mm x partial pathlength factor 6 = 18 cm) to get M.
Channel positions are each participant's MNI.txt (NIRS-SPM cortical points), used as given. Yale
participants whose data channels do not match their positions are skipped: S18-S25 (134 data channels,
128 positions) and S14 (empty MNI.txt).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DATA_ROOT, PPF, hb_to_record, print_summary

SD_DISTANCE = 0.030  # m
TO_MOLAR = 1e-3 / (SD_DISTANCE * 100 * PPF)  # mM*cm -> M, dividing by 18 cm


def read_mni(path):
    """Channel MNI positions in metres, from lines like 'CH01 : -10.0 69.0 22.6'."""
    return np.array([[float(v) for v in line.split(":")[1].split()]
                     for line in open(path, encoding="utf-8-sig") if line.startswith("CH")]) / 1000.0


def read_csv(path):
    """Return HbO, HbR (C, T) and the sampling rate of a Shimadzu export."""
    df = pd.read_csv(path, skiprows=2, header=None, encoding="latin-1")
    signals = df.iloc[:, 3:].to_numpy(float)  # (OxyHb, DeoxyHb, TotalHb) per channel
    sfreq = 1.0 / np.median(np.diff(df.iloc[:, 0].to_numpy(float)))
    return signals[:, 0::3].T, signals[:, 1::3].T, sfreq


def load_hyperscanning_data(main_dir=DATA_ROOT / "hyperscanning", verbose=True):
    """Return ``{recording_name: record}`` for every recording with matching channel positions."""
    outputs = {}
    for site in ["Yale", "UCL"]:
        for sub_dir in sorted(Path(main_dir, site, site).glob("S*")):
            channel_xyz = read_mni(sub_dir / "MNI.txt")
            for path in sorted(p for p in sub_dir.glob("*.csv") if not p.name.startswith("._")):
                hbo, hbr, sfreq = read_csv(path)
                if len(hbo) != len(channel_xyz):
                    if verbose:
                        print(f"Skipped: {path.name} ({len(hbo)} channels, {len(channel_xyz)} positions)")
                    continue
                record = hb_to_record(hbo * TO_MOLAR, hbr * TO_MOLAR, sfreq, channel_xyz,
                                      np.full(len(hbo), SD_DISTANCE))
                record["subject_id"] = f"{site}_{sub_dir.name}"
                outputs[f"{site}_{path.stem}"] = record

    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_hyperscanning_data()
