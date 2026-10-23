"""
von Lühmann, A., Li, X., Gilmore, N., Boas, D. A., Yücel, M. A. (2020). Open access multimodal fNIRS resting state
dataset with and without synthetic hemodynamic responses. Front. Neurosci. 14, 579353.
Data: NITRC project luhmann20synhrf, Dataset I (resting_state_1.zip, manual download).

14 healthy adults, one 5-min resting-state recording each (fixation cross). CW6 (TechEn; 690/830 nm), 50 Hz, 8
sources over the occipital lobe: 26 channels at 30 mm and 8 short channels at 8 mm (the paper says 2 short), all
kept. Only the real recording (resting.snirf) is read; the resting_hrf_*.snirf files are the same data with synthetic
hemodynamic responses added.
Rejected for the main set: no 3D positions. The files store only sourcePos2D/detectorPos2D (AtlasViewer design, mm;
z = 0); the probe file visual_probe_plot.SD is flat as well. Positions are kept as provided.
"""

from pathlib import Path

from loaders.utils import DATA_ROOT, print_summary, snirf_to_record


def load_luhmann_resting_data(main_dir=DATA_ROOT / "luhmann_resting", verbose=True):
    """Return ``{recording_name: record}`` for the real resting-state file of every participant in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("resting_state_1/Subj*/resting.snirf")):
        record = snirf_to_record(path, allow_2d=True)
        record["subject_id"] = path.parent.name                               # Subj33 ... Subj51
        record["coord_frame"] = "2D AtlasViewer probe design (sourcePos2D/detectorPos2D, z = 0), mm"
        outputs[path.parent.name] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_luhmann_resting_data()
