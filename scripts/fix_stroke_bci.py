"""Copy the stroke BCI SNIRF files (Isaev et al. 2024, NITRC) without their empty landmark entries.

The files store landmarkPos2D and landmarkPos3D as empty arrays, which MNE cannot read (TypeError in
read_raw_snirf). Each file is copied and the two empty datasets are deleted from the copy; nothing else changes.

Run once from the repository root:  python -m scripts.fix_stroke_bci
Writes $FNIRS_DATA_ROOT/stroke_bci/<name>.snirf; the original files are not touched.
"""

import shutil

import h5py

from loaders.utils import DATA_ROOT

SRC = DATA_ROOT / "stroke_bci_download/snirf_data"  # files as downloaded (snirf_Raw.zip)
DST = DATA_ROOT / "stroke_bci"                     # fixed copies

DST.mkdir(exist_ok=True)
for path in sorted(SRC.glob("*.snirf")):
    out = DST / path.name
    shutil.copyfile(path, out)                                     # copy, then edit only the copy
    with h5py.File(out, "a") as f:
        probe = f["nirs/probe"]
        for key in ("landmarkPos2D", "landmarkPos3D"):
            if key in probe and probe[key].size == 0:              # empty array: nothing is lost
                del probe[key]
print(f"{len(list(DST.glob('*.snirf')))} files in {DST}")
