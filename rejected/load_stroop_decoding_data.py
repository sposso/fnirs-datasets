"""
Zeng, L., et al. (2026). An fNIRS dataset for Stroop decoding across repeated sessions. Sci. Data 13, 1138,
doi:10.1038/s41597-026-07800-4.
Data: figshare doi:10.6084/M9.FIGSHARE.31839673.V2 ("fNIRS Dataset for Stroop Decoding", StroopDecData.zip).

55 adults performing a block-design Chinese color-word Stroop task three times within about two weeks (test1-3):
165 recordings of ~17 min. LABNIRS (Shimadzu; 780/805/830 nm), 25.6 Hz, 12 sources and 12 detectors, 35 channels
at 30 mm over the prefrontal cortex. Only the HbO/HbR of the Shimadzu software are shared (raw/*.TXT, mM*cm):
divided by 18 cm (30 mm x partial pathlength factor 6) to get M, as for the other Shimadzu datasets.
Author flags: the NIRS_KIT exception_channel of the authors' preprocessed files (prep/*.mat in the zip) marks
channels 17 and 25 of sub306 in test1, and nothing else; these two channels are dropped.
Rejected for the main set: no 3D positions. The probe file ExpBehavData/SD2329-2.SD is a flat 2D grid in mm (z = 0;
stored as 8-bit integers); channel positions are the midpoints of its optodes, kept as provided.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio

from loaders.utils import DATA_ROOT, PPF, hb_to_record, print_summary, read_shimadzu_txt

SD_DISTANCE = 0.03                              # metres (probe file)
TO_MOLAR = 1e-3 / (SD_DISTANCE * 100 * PPF)    # mM*cm -> M, dividing by 18 cm
AUTHOR_EXCLUDED = {("test1", "sub306"): [17, 25]}   # exception_channel of the authors' prep files (1-based)


def load_stroop_decoding_data(main_dir=DATA_ROOT / "stroop_decoding", verbose=True):
    """Return ``{recording_name: record}`` for every raw TXT file in ``main_dir``."""
    root = Path(main_dir) / "StroopDecData"
    probe = sio.loadmat(root / "ExpBehavData" / "SD2329-2.SD", squeeze_me=True, struct_as_record=False)["SD"]
    sources, detectors = probe.SrcPos.astype(float) / 1000, probe.DetPos.astype(float) / 1000   # mm -> m
    outputs = {}
    for path in sorted(root.glob("NirsData/test*/raw/*.TXT")):
        pairs, sfreq, hb = read_shimadzu_txt(path.read_text(errors="ignore"))  # per channel: oxy, deoxy, total
        test = path.parent.parent.name
        keep = [i for i in range(len(pairs)) if i + 1 not in AUTHOR_EXCLUDED.get((test, path.stem), [])]
        xyz = np.array([(sources[s - 1] + detectors[d - 1]) / 2 for s, d in pairs])[keep]
        record = hb_to_record(hb[0::3][keep] * TO_MOLAR, hb[1::3][keep] * TO_MOLAR, sfreq, xyz,
                              np.full(len(keep), SD_DISTANCE))
        record["subject_id"] = path.stem                                       # sub304 ...
        record["coord_frame"] = "flat 2D probe grid (z = 0) of SD2329-2.SD, mm"
        outputs[f"{path.stem}_{test}"] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_stroop_decoding_data()
