"""Convert the HEFMI-ICH fNIRS recordings (figshare 28955456, Homer .nirs files) to SNIRF.

The .nirs files hold raw intensity at three wavelengths (730/808/850 nm). MNE's Beer–Lambert law
takes two, so we keep 730 and 850 nm and drop 808 nm (close to the isosbestic point). Optode
positions are the files' 2D cap layout (cm), which keeps the real 3-cm source-detector separations.
The .nirs files carry no recording date, so the SNIRF files get MNE's anonymised date (2000-01-01).
Our local file names are figshare file IDs plus the session ("54450020_1.nirs"); participants are
recovered from the figshare file list, where each participant's sessions are listed consecutively.
The markers of the stimulus matrix ``s`` are kept as events "marker_<column>" (e.g. marker_9).

Run once from the repository root:  python -m scripts.convert_hefmi_ich
Reads $FNIRS_DATA_ROOT/hefmi_ich_download/source_data/<figshare file id>_<session>.nirs and writes
$FNIRS_DATA_ROOT/hefmi_ich/sub-XX/sub-XX_ses-Y_nirs.snirf; the originals are not touched.
"""

import json
import urllib.request
from datetime import datetime, timezone

import mne
import numpy as np
import scipy.io as sio
from mne_nirs.io import write_raw_snirf

from loaders.utils import DATA_ROOT

SRC = DATA_ROOT / "hefmi_ich_download/source_data"
DST = DATA_ROOT / "hefmi_ich"
FIGSHARE = "https://api.figshare.com/v2/articles/28955456"
KEEP_NM = (730, 850)


def subjects_from_figshare():
    """Return {figshare file id: participant number}; a new participant starts when the session resets."""
    files = json.load(urllib.request.urlopen(FIGSHARE))["files"]
    subjects, subject, previous = {}, 0, np.inf
    for f in files:
        if f["name"].endswith(".nirs"):
            session = int(f["name"].split(".")[0])
            subject += session <= previous
            subjects[str(f["id"])], previous = subject, session
    return subjects


def nirs_to_raw(path):
    """Build an MNE Raw (raw intensity at 730 and 850 nm) from a Homer .nirs file."""
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    sd, ml = m["SD"], np.atleast_2d(m["SD"].MeasList).astype(int)  # columns: source, detector, -, wavelength index
    to_m = {"cm": 0.01, "mm": 0.001}[sd.SpatialUnit]
    src, det = np.atleast_2d(sd.SrcPos) * to_m, np.atleast_2d(sd.DetPos) * to_m
    wl_index = {nm: list(np.atleast_1d(sd.Lambda)).index(nm) + 1 for nm in KEEP_NM}
    rows, names = [], []
    for s, d in dict.fromkeys(map(tuple, ml[:, :2])):  # source-detector pairs, in file order
        for nm in KEEP_NM:
            rows.append(np.flatnonzero((ml[:, 0] == s) & (ml[:, 1] == d) & (ml[:, 3] == wl_index[nm]))[0])
            names.append(f"S{s}_D{d} {nm}")
    info = mne.create_info(names, 1 / np.median(np.diff(m["t"])), "fnirs_cw_amplitude")
    for ch in info["chs"]:
        pair, nm = ch["ch_name"].split(" ")
        s, d = (int(x[1:]) for x in pair.split("_"))
        ch["loc"][3:6], ch["loc"][6:9], ch["loc"][9] = src[s - 1], det[d - 1], float(nm)
        ch["loc"][:3] = (src[s - 1] + det[d - 1]) / 2
    raw = mne.io.RawArray(m["d"][:, rows].T, info, verbose=False)
    raw.set_meas_date(datetime(2000, 1, 1, tzinfo=timezone.utc))  # required by write_raw_snirf
    s = np.asarray(m["s"])                                         # stimulus matrix: samples x markers
    onsets, names = [], []
    for c in range(s.shape[1]):                                    # each marker column becomes events "marker_<column>"
        hits = np.flatnonzero(s[:, c])
        onsets += list(m["t"][hits] - m["t"][0])
        names += [f"marker_{c + 1}"] * len(hits)
    raw.set_annotations(mne.Annotations(onsets, 0.0, names))
    return raw


if __name__ == "__main__":
    subjects = subjects_from_figshare()
    for path in sorted(SRC.glob("*.nirs")):
        file_id, session = path.stem.split("_")
        name = f"sub-{subjects[file_id]:02d}"
        out = DST / name / f"{name}_ses-{session}_nirs.snirf"
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = nirs_to_raw(path)
        raw.info["subject_info"] = {"his_id": name}  # required by write_raw_snirf
        write_raw_snirf(raw, out)
        print(f"{path.name} -> {out.relative_to(DST)} ({raw.n_times / raw.info['sfreq'] / 60:.1f} min)")
