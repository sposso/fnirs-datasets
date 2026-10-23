"""
Peng, K., Yücel, M. A., Steele, S. C., Bittner, E. A., Aasted, C. M., Hoeft, M. A., Lee, A., George, E. E.,
Boas, D. A., Becerra, L., Borsook, D. (2018). Morphine attenuates fNIRS signal associated with painful stimuli in the
medial frontopolar cortex (medial BA 10). Front. Hum. Neurosci. 12, 394.
Data: OpenNeuro ds005777 v1.0.1, doi:10.18112/OPENNEURO.DS005777.V1.0.1 (CC0).

14 right-handed men, two visits each (morphine or placebo, order in participants.tsv), four ~6.5-min runs per visit
(before and 30, 60, 90 min after the pill) of innocuous and noxious electrical stimuli on the left thumb; sub-018 has
an extra run (run-00): 113 recordings. Some sub-009 and sub-010 files are named task-TactilePerception.
CW7 (TechEn; 690/830 nm), 25 Hz, 9 sources, 12 long detectors (24 channels at 30 mm) and 9 short detectors (8 mm):
33 channels, all kept.
Rejected for the main set: no 3D positions. The files store a flat 2D layout in mm (z = 0); coordsystem.json names
CapTrak but gives no landmarks. Positions are kept as provided.
"""

from pathlib import Path

from loaders.utils import DATA_ROOT, print_summary, snirf_to_record


def load_placebo_data(main_dir=DATA_ROOT / "placebo", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*.snirf")):
        record = snirf_to_record(path, allow_2d=True)
        record["subject_id"] = path.name.split("_")[0]                      # sub-002 ... sub-018
        record["coord_frame"] = "flat 2D layout (z = 0), mm; coordsystem.json says CapTrak but has no landmarks"
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_placebo_data()
