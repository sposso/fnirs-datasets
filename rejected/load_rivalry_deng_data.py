"""
Deng, C. (2025). fNIRS dataset: Binocular color rivalry (Red-Green vs. Yellow-Blue) [dataset]. figshare,
doi:10.6084/m9.figshare.29253710 ("Original data.zip", extracted to extracted/); no paper found.

Binocular color rivalry along the red-green (R-G) and yellow-blue (Y-B) directions (10-s stimuli, 10-s rest): 40
recordings of 9-11 min, 20 per direction with the same names (11, 12, 21, ..., 101, 102). The names are read as
participant number + run (an inference; the README only says each file is one subject's session), so subject_id is
the name without its last digit. NirSmart (Huichuang; 730/850 nm), 20 Hz, 6 sources and 5 detectors: 14 channels at
3 cm over the prefrontal cortex; all kept. The Homer .nirs files are read (the .hcx files are the native copies).
Rejected for the main set: no 3D positions. SD.SrcPos/DetPos are a flat 2D grid in cm (z = 0); the README says only
that the layout is based on the 10-20 system. Positions are kept as provided.
"""

from pathlib import Path

from loaders.utils import DATA_ROOT, print_summary, raw_to_record, read_homer_nirs


def load_rivalry_deng_data(main_dir=DATA_ROOT / "rivalry_deng", verbose=True):
    """Return ``{recording_name: record}`` for every .nirs file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("extracted/*/*/*.nirs")):
        record = raw_to_record(read_homer_nirs(path), allow_2d=True)
        record["subject_id"] = f"sub-{path.stem[:-1]}"                         # 11 -> sub-1, 102 -> sub-10
        record["coord_frame"] = "flat 2D grid of the Homer SD (z = 0), cm"
        outputs[f"{path.parent.name}_{path.stem}"] = record                    # e.g. R-G_11
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_rivalry_deng_data()
