"""
Kamat, A., Eastmond, C., Gao, Y., Nemani, A., Yanik, E., Cavuoto, L., Hackett, M., Norfleet, J., Schwaitzberg, S.,
De, S., Intes, X. (2023). Assessment of Surgical Tasks Using Neuroimaging Dataset (ASTaUND). Scientific Data 10, 699.
doi:10.1038/s41597-023-02603-3. Data: figshare doi:10.6084/m9.figshare.c.6429713 ("bids compliant.zip").

Fundamentals of Laparoscopic Surgery pattern-cutting task, in four studies (469 recordings):
- Expert-Novice (14 surgeons) and Learning_Curve (13 residents, 12 training days): TechEn CW6, 690/830 nm, 20-25 Hz,
  8 sources and 16 long detectors forming 25 channels (30-40 mm), plus 8 short detectors (~8 mm, 7 in Learning_Curve).
- Short-Term_tES (12 students) and Long-Term_tES (21 students, 12 days and a follow-up), recorded before and after
  electrical stimulation: NIRx NIRScout, 760/850 nm, 7.81 Hz, the same montage plus three detectors (D17-D19, over
  left and right M1 and SMA), 28 long channels plus 8 short ones (D20-D27).
Positions: only the tES files hold a 3D layout (undocumented frame, cm); scripts/surgical_positions.py moves it into
MNI by its best fit to the NIRx ICBM152 10-5 points. The older studies hold only a flat drawing of the same montage
(paper), so they use the same positions: sources and D1-D16 have the same numbers, and each short detector (D17-D24)
takes the position of the tES short detector at the same source. Beer–Lambert uses the distances of these positions.
Missing samples (zeros, a few files): a missing start or end of up to 1 min is cut off (e.g. the last 22 s of
sub-R10, zero in most channels); gaps of up to 1 s are filled linearly and channels with longer gaps dropped.
"""

from pathlib import Path

import mne
import numpy as np

from .utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

STUDIES = ("Expert-Novice", "Learning_Curve", "Short-Term_tES", "Long-Term_tES")
OLDER_STUDIES = ("Expert-Novice", "Learning_Curve")              # flat drawing only, montage without D17-D19

# Optode MNI positions in mm (scripts/surgical_positions.py): the tES layout moved into MNI. D2 and D3 lie ~20 mm
# inside the template head in the authors' layout and are kept as given.
OPTODE_MNI = {
    # Sources
    "S1": (-29.5, 94.2, -10.7), "S2": (0.2, 86.7, -6.4), "S3": (29.9, 91.9, -10.3), "S4": (-68.4, -0.6, 61.3),
    "S5": (-27.8, -8.7, 92.6), "S6": (31.0, -10.6, 91.7), "S7": (69.9, 1.9, 59.0), "S8": (0.7, 29.9, 89.5),
    # Long detectors of both montages
    "D1": (-46.2, 81.4, -22.5), "D2": (-14.2, 68.7, -9.0), "D3": (13.1, 67.8, -8.8), "D4": (45.6, 77.9, -21.7),
    "D5": (-73.0, 28.8, 37.4), "D6": (-80.9, -6.7, 41.6), "D7": (-54.3, 10.6, 72.5), "D8": (-58.2, -28.1, 76.1),
    "D9": (0.5, 10.2, 95.0), "D10": (0.1, -30.3, 99.0), "D11": (55.0, 10.7, 72.3), "D12": (58.5, -28.0, 76.1),
    "D13": (73.7, 28.7, 37.2), "D14": (81.2, -6.2, 41.6), "D15": (-16.5, 48.6, 79.5), "D16": (18.0, 48.6, 79.4),
    # Long detectors added in the tES studies (left M1, right M1, SMA)
    "D17": (-77.0, -32.6, 52.6), "D18": (78.8, -21.4, 52.0), "D19": (1.0, 66.0, 71.0),
    # Short detectors of the tES studies, next to S1-S8
    "D20": (-21.6, 94.7, -7.8), "D21": (7.4, 86.4, -6.5), "D22": (36.6, 92.3, -13.5), "D23": (-64.9, 7.9, 62.3),
    "D24": (-23.8, -1.8, 93.0), "D25": (34.8, -18.6, 90.8), "D26": (73.2, -5.4, 57.2), "D27": (8.7, 30.8, 88.8),
}
OPTODE_XYZ = {name: np.array(xyz) / 1000.0 for name, xyz in OPTODE_MNI.items()}  # mm -> m


def older_study_positions(raw):
    """Optode positions for an older-study recording: each short detector (numbered above 16) takes the position
    of the tES short detector at the same source (S1 -> D20, ..., S8 -> D27)."""
    xyz = dict(OPTODE_XYZ)
    for name in raw.ch_names:
        source, detector = name.split(" ")[0].split("_")          # "S1_D17 690" -> S1, D17
        if int(detector[1:]) > 16:
            xyz[detector] = OPTODE_XYZ[f"D{19 + int(source[1:])}"]
    return xyz


def load_surgical_data(main_dir=DATA_ROOT / "surgical", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file of the four studies in ``main_dir``."""
    outputs = {}
    for study in STUDIES:
        for path in sorted(Path(main_dir, study).glob("sub-*/**/*.snirf")):
            raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
            sfreq = raw.info["sfreq"]
            raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
            optode_xyz = older_study_positions(raw) if study in OLDER_STUDIES else OPTODE_XYZ
            record = raw_to_record(raw, optode_xyz=optode_xyz, fix_file_positions=True)
            record["subject_id"] = f"{study}_{path.name.split('_')[0]}"  # subject IDs repeat across studies
            outputs[f"{study}_{path.stem}"] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_surgical_data()
