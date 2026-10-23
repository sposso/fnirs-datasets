"""Write readable copies of the two sub-400 left-hemisphere resting files of the inclusion dataset (ds006377).

sub-400_task-001RestingLH_run-1 and sub-400_task-003RestingLH_run-2 have an empty landmark table, which
MNE cannot read, and scrambled optode positions. Their channel list, wavelengths and source-detector
distances are identical to every other left-hemisphere file (same cap and probe for all participants), so
the copies get the standard left-hemisphere probe (positions and landmarks) of sub-402's matching file.

Run once from the repository root:  python -m scripts.fix_inclusion_sub400
Writes $FNIRS_DATA_ROOT/inclusion_fixed/<file name>; the originals are not touched.
"""

import shutil

import h5py

from loaders.utils import DATA_ROOT

SRC = DATA_ROOT / "inclusion"
DST = DATA_ROOT / "inclusion_fixed"
FILES = ["sub-400_task-001RestingLH_run-1_nirs.snirf", "sub-400_task-003RestingLH_run-2_nirs.snirf"]
REFERENCE = SRC / "sub-402/nirs/sub-402_task-001RestingLH_run-1_nirs.snirf"

if __name__ == "__main__":
    DST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        out = DST / name
        shutil.copy(SRC / "sub-400/nirs" / name, out)
        with h5py.File(out, "r+") as dst, h5py.File(REFERENCE, "r") as ref:
            del dst["nirs/probe"]
            ref.copy("nirs/probe", dst["nirs"])  # same probe: wavelengths, optode and landmark positions
        print(f"wrote {out}")
