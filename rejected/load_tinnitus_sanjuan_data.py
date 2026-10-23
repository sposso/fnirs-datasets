"""
San Juan, J., Hu, X.-S., Issa, M., Bisconti, S., Kovelman, I., Kileny, P., Basura, G. (2017). Tinnitus alters
resting state functional connectivity (RSFC) in human auditory and non-auditory brain regions as measured by
functional near-infrared spectroscopy (fNIRS). PLoS ONE 12(6), e0179150.
Data: Harvard Dataverse doi:10.7910/DVN/ZNZZBV (18 Homer .nirs files).

18 adults (8 controls, 10 with tinnitus), one resting-state recording of ~18 min each. CW6 (TechEn; 690/830 nm),
20 Hz (BZZ004: 50 Hz), 44 channels at 29-30 mm over both hemispheres; all kept. No measurement is deactivated
(MeasListAct and MeasListMask are all 1; MeasListVis only hides plots in Homer).
Rejected for the main set: no 3D positions. SD.SrcPos/DetPos are a flat 2D layout in mm (z = 0); the authors'
digitized positions are not shared. Positions are kept as provided.
"""

from pathlib import Path

from loaders.utils import DATA_ROOT, print_summary, raw_to_record, read_homer_nirs


def load_tinnitus_sanjuan_data(main_dir=DATA_ROOT / "tinnitus_sanjuan", verbose=True):
    """Return ``{recording_name: record}`` for every .nirs file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("*.nirs")):
        record = raw_to_record(read_homer_nirs(path), allow_2d=True)
        record["subject_id"] = path.stem                                       # BZZ003 ... BZZ028
        record["coord_frame"] = "flat 2D layout of the Homer SD (z = 0), mm"
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_tinnitus_sanjuan_data()
