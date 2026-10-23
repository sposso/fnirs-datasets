"""Make MNE-readable copies of the Cambridge HD-DOT datasets: resting state (OpenNeuro ds007719 v1.0.0) and visual
task (OpenNeuro ds007714 v1.0.0), from the same study.

The SNIRF files lack required fields (formatVersion and the metaDataTags TimeUnit, MeasurementDate,
MeasurementTime, SubjectID and FrequencyUnit) and give LengthUnit as "millimeter", so MNE cannot read them.
The copies add these fields (time in s, as the 0.08-s and 0.2-s steps show; date and time unknown) and set
LengthUnit to "mm"; the data and positions are not changed. The BIDS sidecar files are copied as they are.
Files that cannot be opened (sub-s45 of the visual task is truncated in the upload) are not copied.

Run once from the repository root:  python -m scripts.fix_cambridge_hddot [resting|visual]   (default: both)
Reads $FNIRS_DATA_ROOT/<dataset>_download/ and writes $FNIRS_DATA_ROOT/<dataset>/; the original files are not touched; existing copies are
left as they are.
"""

import shutil
import sys

import h5py

from loaders.utils import DATA_ROOT

DATASETS = {"resting": "cambridge_hddot", "visual": "cambridge_visual"}  # folder of the fixed copies
MISSING_TAGS = {"TimeUnit": "s", "MeasurementDate": "unknown", "MeasurementTime": "unknown", "FrequencyUnit": "Hz"}


def fix(name):
    src, dst = DATA_ROOT / f"{name}_download", DATA_ROOT / name             # original download, fixed copies
    if dst.exists():
        print(f"{dst} exists, left as it is")
        return
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git*", ".datalad"))
    for path in sorted(dst.glob("sub-*/nirs/*_nirs.snirf")):
        try:
            f = h5py.File(path, "r+")
        except OSError:                                                      # truncated upload
            print(f"cannot open {path.name}: not copied")
            path.unlink()
            continue
        with f:
            tags = f["nirs/metaDataTags"]
            del tags["LengthUnit"]
            tags["LengthUnit"] = "mm"
            for key, value in MISSING_TAGS.items():
                tags[key] = value
            tags["SubjectID"] = path.name.split("_")[0]   # e.g. sub-s04
            f["formatVersion"] = "1.0"
    print(f"fixed {len(list(dst.glob('sub-*/nirs/*_nirs.snirf')))} SNIRF files in {dst}")


if __name__ == "__main__":
    for name in sys.argv[1:] or DATASETS:
        fix(DATASETS[name])
