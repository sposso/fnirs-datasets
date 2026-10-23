"""
fNIRS resting state with motion artifacts - Dataset II (openfnirs; Yücel laboratory). NITRC project yucel14motion,
file motion_artifacts_2.zip from the fNIRS Research Database (https://www.nitrc.org/frs/?group_id=1071), manual
download. Same laboratory as Yücel et al. (2014), NeuroImage 85, 192-201 (Dataset I = load_motion_data).

5 resting-state recordings with motion artifacts (rest5, rest15, rest17, rest18, rest20), 3.7-7.6 min each, stored as
both Homer .nirs and SNIRF with identical data; the .nirs files are read because only they keep the authors' channel
deactivations (SD.MeasListAct = 0: 8 measurements in rest15, 6 in rest17, 1 in rest20), whose source-detector pairs
are dropped. CW6 (TechEn; 690/830 nm), 50 Hz, 15 sources and 31 detectors: 51 channels (36 at 30 mm, 15 short at
8 mm), the probe of the thermal dataset. Zero or negative intensities (rest15) follow the shared rule (fill_or_drop).
Rejected for the main set: no 3D positions. SD.SrcPos/DetPos are a flat 2D layout in mm (z = 0; the SNIRF files hold
only sourcePos2D/detectorPos2D), without landmarks. Positions are kept as provided.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio

from loaders.utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record, read_homer_nirs


def load_motion2_data(main_dir=DATA_ROOT / "motion2", verbose=True):
    """Return ``{recording_name: record}`` for every .nirs file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("motion_artifacts_2/*.nirs")):
        raw = read_homer_nirs(path)
        probe = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["SD"]
        deactivated = {f"S{s}_D{d}" for (s, d, _, _), active in zip(np.asarray(probe.MeasList).astype(int),
                                                                   np.ravel(probe.MeasListAct)) if not active}
        raw.drop_channels([ch for ch in raw.ch_names if ch.split(" ")[0] in deactivated])
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw, allow_2d=True)
        record["subject_id"] = f"sub-{path.stem[4:]}"                           # rest5 -> sub-5
        record["coord_frame"] = "flat 2D layout of the Homer SD (z = 0), mm, no landmarks"
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_motion2_data()
