"""
Spapé, M., Mäkelä, K., Ruotsalo, T. (2024). NEMO: A database for emotion analysis using functional near-infrared
spectroscopy. IEEE Trans. Affect. Comput. 15(3), 1166-1177.
Data: Zenodo doi:10.5281/ZENODO.8333978 (BIDS, also OSF pd9rv).

31 adults viewing emotional pictures and imagining affective scenarios in alternating blocks: one session of 29-67 min
each (2 stopped halfway). Brite-24 (Artinis; 756/853 nm), 50 Hz, 10 sources and 8 detectors: 24 channels at 30 mm
over the forehead. The files hold optical density. The empe and afim files of each participant contain the same
recording (identical data, different events): only empe is read.
Measurements that are zero for the whole recording (21 files, as the authors note) are missing data: their
source-detector pairs are dropped.
Rejected for the main set: no measured positions. The files hold a flat 3 x 6 grid (3 cm spacing, z = 0, m) applied by
the authors, with fsaverage template fiducials; coordsystem.json names CapTrak. Positions are kept as provided.
"""

from pathlib import Path

import mne

from loaders.utils import DATA_ROOT, print_summary, raw_to_record


def load_nemo_data(main_dir=DATA_ROOT / "nemo", verbose=True):
    """Return ``{recording_name: record}`` for the empe SNIRF file of every participant in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*_task-empe_nirs.snirf")):
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        empty = {ch.split(" ")[0] for ch, x in zip(raw.ch_names, raw.get_data()) if not x.any()}
        record = raw_to_record(raw, drop_pairs=empty, allow_2d=True)
        record["subject_id"] = path.name.split("_")[0]                      # sub-101 ... sub-1xx
        record["coord_frame"] = "flat 3 x 6 grid (z = 0), m, applied by the authors; template fiducials (CapTrak)"
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_nemo_data()
