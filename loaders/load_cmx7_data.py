"""
Ajra, Z., Vergotte, G., Perrey, S., Evra, L., Pla, S., Dray, G., Montmain, J., Xu, B. (2026).
Multimodal dataset from the CMx7-MM Experiment. OpenNeuro ds007554. doi:10.18112/openneuro.ds007554.v1.0.0

30 participants x 3 sessions x 7 cognitive-motor tasks of ~3.6 min (519 fNIRS recordings),
sampled at 10 Hz (sub-001 to sub-020) or 50 Hz (sub-021 to sub-030).
16 channels (16 sources x 4 detectors, 760/850 nm) over the forehead and the motor cortex;
no short channels.
The positions in the files are the NIRx ICBM152 (MNI) template points of 10-5 labels, but they are
stored in cm while the files say mm, and in 67 of the 74 activemotor files the motor-cortex optodes
carry the forehead coordinates. Every file therefore gets the template position of each optode's
10-5 label before processing.
"""

from pathlib import Path

import numpy as np

from .utils import DATA_ROOT, print_summary, snirf_to_record

# optode: (10-5 label, MNI position in mm on the NIRx ICBM152 template)
OPTODES = {
    # Sources: forehead (S1-S8) and motor cortex (S9-S16)
    "S1": ("F10h", (70.573, 42.152, -38.921)),
    "S2": ("FFT8h", (74.242, 32.900, -1.199)),
    "S3": ("AF4", (35.272, 78.970, 16.408)),
    "S4": ("Fp2", (29.184, 85.158, -13.797)),
    "S5": ("Fp1", (-29.230, 85.182, -14.189)),
    "S6": ("AF3", (-35.609, 78.819, 16.092)),
    "S7": ("F7h", (-68.536, 46.263, -4.366)),
    "S8": ("F9h", (-71.147, 41.230, -39.636)),
    "S9": ("FC6h", (71.699, 22.827, 37.174)),
    "S10": ("CP6h", (78.498, -46.421, 48.423)),
    "S11": ("CP4h", (55.491, -47.322, 81.614)),
    "S12": ("FC4h", (50.188, 26.867, 67.163)),
    "S13": ("FC3h", (-50.962, 25.976, 66.860)),
    "S14": ("CP3h", (-55.921, -48.229, 81.020)),
    "S15": ("CP5h", (-78.435, -47.700, 47.624)),
    "S16": ("FC5h", (-72.490, 21.549, 36.688)),
    # Detectors
    "D1": ("AF8", (54.021, 69.084, -17.587)),
    "D2": ("AF7", (-54.498, 68.788, -18.293)),
    "D3": ("C4", (69.701, -9.782, 63.617)),
    "D4": ("C3", (-70.634, -11.463, 63.077)),
}
OPTODE_XYZ = {name: np.array(xyz) / 1000.0 for name, (_, xyz) in OPTODES.items()}  # mm -> m


def load_cmx7_data(main_dir=DATA_ROOT / "cmx7", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*.snirf")):
        record = snirf_to_record(path, optode_xyz=OPTODE_XYZ, fix_file_positions=True)
        record["subject_id"] = path.parts[-4]  # sub-XXX
        outputs[path.stem] = record

    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_cmx7_data()
