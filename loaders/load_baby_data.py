"""
Blanco, B., Molnar, M., Carreiras, M., Caballero-Gaudes, C. (2022). Open access dataset of task-free
hemodynamic activity in 4-month-old infants during sleep using fNIRS. Scientific Data 9, 102.
doi:10.1038/s41597-022-01210-y

104 infants, one NIRScout recording each (10-26 min, 8.93 Hz) during natural sleep. The 5 infants the
authors discarded for insufficient data quality (OSF BCBL_RS4_participant-info.csv: "Discarded") are
skipped, leaving 99.
52 channels (16 sources x 24 detectors, 760/850 nm, 20-45 mm). The 6 occipital channels are dropped,
as in the authors' analysis ("particularly prone to contain signal artifacts"; Blanco et al. 2021,
Neurophotonics 8(2), 025011), leaving 46.
No 3D positions were digitized: the files hold a flat 2D layout (z = 0) that only encodes the real
source-detector distances. Optodes were placed at 10-20 positions (OSF diagram
BCBL_RS4_optode-setup_reordered.pdf, p. 3), so each optode gets the MNI position of its 10-20 label on
the NIRx ICBM152 template, the same template points the Appearance and Athlete optodes sit on.
Beer–Lambert and ``sd_distances`` use the file's real infant distances.
"""

from pathlib import Path

import numpy as np

from .utils import DATA_ROOT, print_summary, snirf_to_record

# Infants discarded by the authors for insufficient data quality (OSF participant-info.csv)
DISCARDED = {"sub-4310", "sub-4348", "sub-4350", "sub-4550", "sub-4637"}

# Occipital channels discarded by the authors (optodes S9, S10, D12-D16 are only used by these)
DROP_PAIRS = {"S9_D12", "S9_D13", "S9_D14", "S10_D14", "S10_D15", "S10_D16"}

# optode: (10-20 label from the OSF diagram, MNI position in mm on the NIRx ICBM152 template)
OPTODES = {
    # Sources
    "S1": ("F4", (51.167, 56.349, 37.498)),
    "S2": ("F3", (-51.830, 55.753, 37.164)),
    "S3": ("C3", (-70.634, -11.463, 63.077)),
    "S4": ("FC5", (-77.435, 19.001, 18.879)),
    "S5": ("F7", (-70.550, 42.879, -19.263)),
    "S6": ("CP5", (-82.542, -47.309, 27.800)),
    "S7": ("T7", (-84.830, -16.239, -16.314)),
    "S8": ("P7", (-73.902, -75.644, -7.512)),
    "S11": ("C4", (69.701, -9.782, 63.617)),
    "S12": ("FC6", (76.997, 20.379, 19.465)),
    "S13": ("F8", (70.103, 43.768, -18.472)),
    "S14": ("CP6", (82.818, -45.959, 28.546)),
    "S15": ("T8", (84.621, -14.204, -15.575)),
    "S16": ("P8", (74.377, -74.619, -6.956)),
    # Detectors
    "D1": ("AF4", (35.272, 78.970, 16.408)),
    "D2": ("Fz", (-0.235, 62.434, 65.513)),
    "D3": ("AF3", (-35.609, 78.819, 16.092)),
    "D4": ("FC3", (-63.100, 23.879, 52.682)),
    "D5": ("F5", (-64.987, 49.590, 10.244)),
    "D6": ("CP3", (-69.286, -48.011, 65.668)),
    "D7": ("C5", (-84.098, -13.846, 24.756)),
    "D8": ("FT7", (-79.225, 13.769, -18.190)),
    "D9": ("P5", (-70.586, -78.617, 25.663)),
    "D10": ("TP7", (-84.057, -46.448, -12.942)),
    "D11": ("PO7", (-56.117, -100.316, -0.797)),
    "D17": ("FC4", (62.315, 24.986, 53.080)),
    "D18": ("F6", (64.380, 50.370, 10.803)),
    "D19": ("CP4", (69.044, -46.885, 66.427)),
    "D20": ("C6", (83.687, -11.831, 25.487)),
    "D21": ("FT8", (78.937, 15.204, -17.416)),
    "D22": ("P6", (71.064, -77.654, 26.372)),
    "D23": ("TP8", (84.505, -45.070, -12.301)),
    "D24": ("PO8", (56.669, -99.838, -0.394)),
}
OPTODE_XYZ = {name: np.array(xyz) / 1000.0 for name, (_, xyz) in OPTODES.items()}  # mm -> m


def load_baby_data(main_dir=DATA_ROOT / "baby_data", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*.snirf")):
        if path.parts[-3] in DISCARDED:
            continue
        record = snirf_to_record(path, drop_pairs=DROP_PAIRS, optode_xyz=OPTODE_XYZ)
        record["subject_id"] = path.parts[-3]  # sub-XXXX
        outputs[path.stem] = record

    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_baby_data()
