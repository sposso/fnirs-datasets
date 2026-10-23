"""
Ning, M., Duwadi, S., Yücel, M. A., von Lühmann, A., Boas, D. A., Sen, K. (2024). fNIRS dataset during complex scene
analysis. Front. Hum. Neurosci. 18, 1329086.
Data: OpenNeuro ds004830 v1.0.1 ("Spatial Attention Decoding using fNIRS During Complex Scene Analysis"),
doi:10.18112/OPENNEURO.DS004830.V1.0.1.

12 adults attending to one of three talkers (audio-visual cocktail-party paradigm): 19 recordings of 2-59 min.
CW6 (TechEn; 690/830 nm), 50 Hz. Standard probe: 12 sources, 17 long and 6 short detectors, 30 long (28-30 mm) and
6 short (8 mm) channels; sub-08: 34 long and 8 short channels. All kept; no author flags.
Rejected for the main set: no 3D positions. The files store the 2D AtlasViewer probe design (z = 0, mm; the source-
detector distances are correct); coordsystem.json names CapTrak but gives no landmarks. Positions are kept as provided.
"""

from pathlib import Path

from loaders.utils import DATA_ROOT, print_summary, snirf_to_record


def load_spatial_attention_data(main_dir=DATA_ROOT / "spatial_attention", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*.snirf")):
        record = snirf_to_record(path, allow_2d=True)
        record["subject_id"] = path.name.split("_")[0]                      # sub-08 ... sub-25
        record["coord_frame"] = "2D AtlasViewer probe design (z = 0), mm; coordsystem.json says CapTrak, no landmarks"
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_spatial_attention_data()
