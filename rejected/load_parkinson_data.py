"""
Guevara, E., Rivas-Ruvalcaba, F. J., Kolosovas-Machuca, E. S., Ramírez-Elías, M., Díaz de León Zapata, R.,
Ramirez-GarciaLuna, J. L., Rodríguez-Leyva, I. (2024). Parkinson's disease patients show delayed hemodynamic changes
in primary motor cortex in fine motor tasks and decreased resting-state interhemispheric functional connectivity: a
functional near-infrared spectroscopy study. Neurophotonics 11(2), 025004.
Data: Zenodo doi:10.5281/ZENODO.7966830 (CC BY 4.0); the local copy equals the zip (120 SNIRF files, identical).

20 patients with Parkinson's disease (PD01-PD20) and 20 controls (Control01-Control20), three recordings each: rest
(6 min, eyes closed), finger tapping (~11 min) and walking (2 min): 120 recordings. The old project's test copy
walking/Control/Control02/3_walking_units_cm_testcopy.snirf (not in the zip) is skipped.
Brite MKII (Artinis; 757/843 nm), 25 Hz (17 files at 50 or 75 Hz), 10 sources and 8 detectors: 20 long and 2 short
channels over both motor cortices.
Missing data: every tapping and walking file has samples at exactly 2^-16 (the smallest 16-bit step) on all channels at
once, ~20 times below any other value: device dropouts. They are treated as missing and follow the shared rule
(fill_or_drop): no recording is lost; Control07 and Control09 lose 4 and 1 channels.
Rejected for the main set: no 3D positions. The files store a flat 2D drawing (z = 0) declared in mm whose distances
read 3.0 and 1.0 (i.e. cm), without landmarks; positions are kept as provided, in the declared unit. Beer-Lambert uses
the paper's distances: 3 cm for the long channels and 1.5 cm for the short ones (sd_distance), as for the motion data.
"""

from pathlib import Path

import mne
import numpy as np

from loaders.utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

LONG, SHORT = 0.03, 0.015       # metres (paper)
DROPOUT = 2.0 ** -16            # device value of a lost sample


def load_parkinson_data(main_dir=DATA_ROOT / "parkinson", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("*/*/*/*.snirf")):
        if "testcopy" in path.name:                                           # the old project's copy
            continue
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        raw.apply_function(lambda x: np.where(np.isclose(x, DROPOUT, rtol=1e-3), np.nan, x))
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        drawn = mne.preprocessing.nirs.source_detector_distances(raw.info)    # 0.003 or 0.001 (declared mm)
        lengths = {ch.split(" ")[0]: SHORT if d < 0.002 else LONG for ch, d in zip(raw.ch_names, drawn)}
        record = raw_to_record(raw, sd_distance=lengths, allow_2d=True)
        record["subject_id"] = path.parent.name                               # Control01 ... PD20
        record["coord_frame"] = "flat 2D drawing (z = 0), declared mm (drawn in cm), no landmarks"
        outputs[f"{path.parent.name}_{path.stem}"] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_parkinson_data()
