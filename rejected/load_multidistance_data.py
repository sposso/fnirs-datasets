"""
Gao, Y., Rogers, D., von Lühmann, A., Ortega-Martinez, A., Boas, D. A., Yücel, M. A. (2023). Short-separation
regression incorporated diffuse optical tomography image reconstruction modeling for high-density functional
near-infrared spectroscopy. Neurophotonics 10(2), 025007.
Data: OpenNeuro ds007420 v1.0.5 ("Open-access Multi-Distance fNIRS Dataset Generated During Ball-Squeezing and
Motion-Artifact Induction Tasks"), doi:10.18112/openneuro.ds007420.v1.0.5.

12 adults: 60 recordings (ball squeezing 36, motion 12, rest 12), 7.52 h. Two cascaded NIRSport2 (NIRx; 760/850 nm),
8.72 or 11.63 Hz, 14 sources and 32 detectors: the 100 designed channels (17-43 mm), all kept; no author flags
(channels.tsv has no status; the quality report in derivatives is not a flag).
Rejected for the main set: undocumented position frame. The files store 3D optode positions in mm, but
coordsystem.json calls them "2D coordinates from AtlasViewer generated SD file" (frame "Other") and gives no
landmarks. Positions are kept as provided.
"""

from pathlib import Path

from loaders.utils import DATA_ROOT, print_summary, snirf_to_record


def load_multidistance_data(main_dir=DATA_ROOT / "multidistance", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*.snirf")):
        record = snirf_to_record(path)
        record["subject_id"] = path.name.split("_")[0]                      # sub-170 ...
        record["coord_frame"] = "3D positions in an undocumented frame ('Other'), mm, no landmarks"
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_multidistance_data()
