"""
Bálint, A., Wimmer, W., Rummel, C., Caversaccio, M., Weder, S. (2025). An fNIRS dataset for Multimodal Speech
Comprehension in Normal Hearing Individuals and Cochlear Implant Users. Scientific Data 12, 1305.
doi:10.1038/s41597-025-05654-w. Data: Dryad doi:10.5061/dryad.crjdfn3g9 (snirf_local-bids.zip, CC0).

72 adults (46 cochlear implant users, sub-S*; 26 normal-hearing controls, sub-C*), one recording each (26-29 min):
sentences of the German Matrix Sentence Test in quiet, in noise, audiovisual and visual only (lipreading), in ten
counterbalanced blocks with comprehension questions. Shimadzu FOIRE-3000, 780/805/830 nm, 14.29 Hz: 16 sources and
16 detectors forming 39 channels (3 of them short) over the prefrontal, temporal and occipital cortices.
The files hold absorbance (log10; the log10 extinction coefficients reproduce Shimadzu's 3-wavelength formula given in
the paper), although the SNIRF labels it as raw intensity, at three wavelengths, which MNE cannot read. The 780 and
830 nm signals are used as optical density (absorbance x ln 10, MNE's natural-log convention) with MNE's Beer–Lambert
law, as for the other datasets; 805 nm is not used.
Channels set to NaN by the authors (probes removed for the implant's transmitter, 0-8 per recording; the same optodes
have no position) are dropped.
Positions: each participant's probe positions from a 3D scan, aligned by the authors to MNE's 10-05 template
(fsaverage MRI coordinates, i.e. MNI), used as given (coordsystem.json wrongly says CapTrak); 23 participants whose
scan failed have the positions of a participant with a matching head size.
Events: the BIDS events.tsv (AStim, ANStim, AVStim, VStim: 13-s sentence blocks in quiet, in noise, audiovisual and
visual only; plus startorstop, Question and answers). Downstream task (GeoNIRS): normal hearing vs CI, one sample per
sentence block, labelled with the participant's group.
"""

from pathlib import Path

import h5py
import mne
import numpy as np

from ..utils import DATA_ROOT, print_summary, raw_to_record
from . import bids_events, epochs

CONDITION = (0.0, 13.0)  # start after the event and duration (s); paper: "each 13-second long stimulus"

WAVELENGTHS = (780, 830)  # nm used (805 nm dropped: MNE's Beer–Lambert takes two wavelengths)


def read_two_wavelengths(path):
    """MNE Raw of optical density at 780 and 830 nm, with the optode positions, from one Shimadzu SNIRF file."""
    with h5py.File(path, "r") as f:
        nirs = f["nirs"]
        time = nirs["data1/time"][()]
        absorbance = nirs["data1/dataTimeSeries"][()]                     # time x signals, log10 absorbance
        wavelengths = nirs["probe/wavelengths"][()]
        sources = nirs["probe/sourcePos3D"][()][:, :3] / 1000.0           # mm -> m
        detectors = nirs["probe/detectorPos3D"][()][:, :3] / 1000.0
        signals = {}                                                       # (source, detector, wavelength) -> column
        for column in range(absorbance.shape[1]):
            entry = nirs[f"data1/measurementList{column + 1}"]
            index = [int(np.ravel(entry[key][()])[0]) for key in ("sourceIndex", "detectorIndex", "wavelengthIndex")]
            signals[(index[0], index[1], int(wavelengths[index[2] - 1]))] = column
    pairs = sorted({(s, d) for s, d, _ in signals})
    keys = [(s, d, w) for s, d in pairs for w in WAVELENGTHS]              # each pair's two wavelengths side by side
    info = mne.create_info([f"S{s}_D{d} {w}" for s, d, w in keys], (len(time) - 1) / (time[-1] - time[0]), "fnirs_od")
    for ch, (s, d, w) in zip(info["chs"], keys):
        ch["loc"][3:6], ch["loc"][6:9] = sources[s - 1], detectors[d - 1]
        ch["loc"][:3] = (sources[s - 1] + detectors[d - 1]) / 2
        ch["loc"][9] = w
    data = absorbance[:, [signals[k] for k in keys]].T * np.log(10)        # log10 absorbance -> natural-log OD
    return mne.io.RawArray(data, info, verbose=False)


def load_speech_comprehension_data(main_dir=DATA_ROOT / "speech_comprehension", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*_nirs.snirf")):
        raw = read_two_wavelengths(path)
        raw.set_annotations(bids_events(path))
        flagged = {name.split(" ")[0] for name, x in zip(raw.ch_names, raw.get_data()) if np.isnan(x).all()}
        record = raw_to_record(raw, drop_pairs=flagged)                    # drop the authors' NaN channels
        record["subject_id"] = path.name.split("_")[0]                     # e.g. sub-C01
        record["group"] = "normal hearing" if record["subject_id"].startswith("sub-C") else "CI"
        record["event_labels"] = {name: record["group"] for name in ("AStim", "ANStim", "AVStim", "VStim")}
        record["epochs"], record["labels"] = epochs(record, *CONDITION)
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_speech_comprehension_data()
