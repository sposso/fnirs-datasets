"""
Mutlu, M.C., Canbeyli, R., Saybaşılı, H. (2023). Functional near-infrared spectroscopy shows that object relative
clauses are more difficult to process than subject relative clauses in Turkish. European Journal of Neuroscience 57(6),
951-961. doi:10.1111/ejn.15930.
Data: figshare doi:10.6084/m9.figshare.21923175 (v1, CC BY 4.0; MD5 checked): one raw SNIRF file per participant
(converted from .nirs without preprocessing).

14 native Turkish speakers answered yes/no questions about 24 sentences containing a subject- or object-relative
clause (one recording of 6.9-7.4 min each); the file of subject 6 was damaged and is not shared: 13 recordings.
NIRSport (NIRx; 760/850 nm, 7.81 Hz), 8 sources and 7 detectors over the prefrontal cortex, 22 channels (3 cm in the
paper). The authors excluded channels outside an 80-140 dB range in their analysis, but these are not listed in the
dataset; all channels are kept.
Positions: the files hold optode positions identical for all participants, in an undocumented frame without
landmarks. The channel MNI coordinates of the same probe (NIRS_SPM registration) are given in Table 1 of
Mutlu et al. (2020, Front. Hum. Neurosci. 14:252, doi:10.3389/fnhum.2020.00252), from which the 2023 paper takes
its probe figure. The channel midpoints of the files match that table, in the files' channel order, up to a rigid
transform (mean residual 2.2 mm, max 3.6 mm; table rounded to 1 mm), so the table is used as the channel positions.
Beer–Lambert uses the stored source-detector distances (16-42 mm).
"""

from pathlib import Path

import mne
import numpy as np

from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

# Mutlu et al. (2020), Table 1: MNI coordinates (mm) of channels 1-22 (= the files' channel order)
CHANNELS = {"S1_D1": (29, 46, 45), "S1_D3": (41, 49, 32), "S2_D1": (11, 48, 52), "S2_D2": (-12, 48, 50),
            "S2_D4": (-4, 57, 41), "S3_D2": (-33, 44, 40), "S3_D5": (-42, 48, 28), "S4_D1": (20, 56, 41),
            "S4_D3": (31, 60, 27), "S4_D4": (11, 66, 31), "S4_D6": (23, 71, 14), "S5_D2": (-27, 53, 37),
            "S5_D4": (-16, 65, 28), "S5_D5": (-36, 56, 25), "S5_D7": (-27, 68, 10), "S6_D3": (44, 60, 6),
            "S6_D6": (34, 67, -6), "S7_D4": (-2, 69, 11), "S7_D6": (13, 73, -4), "S7_D7": (-13, 73, -5),
            "S8_D5": (-43, 57, 5), "S8_D7": (-32, 66, -6)}
CHANNEL_POS = {pair: np.array(xyz) / 1000.0 for pair, xyz in CHANNELS.items()}


def load_turkish_relative_data(main_dir=DATA_ROOT / "turkish_relative", verbose=True):
    """Return ``{recording_name: record}`` for every participant folder (``S1`` ... ``S14``) in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("S*/Data.snirf"), key=lambda p: int(p.parent.name[1:])):
        raw = mne.io.read_raw_snirf(path, verbose=False).load_data(verbose=False)
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        if raw is None:
            continue
        record = raw_to_record(raw, channel_pos=CHANNEL_POS)
        record["subject_id"] = path.parent.name                             # S1 ... S14 (no S6)
        outputs[path.parent.name] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_turkish_relative_data()
