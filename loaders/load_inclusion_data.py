"""
Yücel, M. A., Anderson, J. E., Rogers, D., et al. (2025). Quantifying the impact of hair and skin
characteristics on fNIRS signal quality for enhanced inclusivity. Nature Human Behaviour 9, 2651-2668.
doi:10.1038/s41562-025-02274-7. OpenNeuro ds006377.

115 adults: two ~3-min resting runs (after quick and after careful capping) and one ~6-min left/right-hand
ball-squeezing run. Each hemisphere was recorded at the same time by its own NIRSport2 (760/850 nm,
10.17 Hz), so every run has a left (LH) and a right (RH) file: 690 files, kept as separate recordings.
Per hemisphere: 8 sources and 16 detectors forming 26 channels (18 long at 15-34 mm, 8 short at ~8 mm)
over the forehead, the side and the back of the head; all channels are kept.
Optode positions are the NIRx ICBM152 (MNI) template positions of the cap, the same for everyone.
Two sub-400 files cannot be read by MNE and have scrambled positions; they are read from fixed copies
(scripts/fix_inclusion_sub400.py) in ``fixed_dir``. In sub-343 and sub-457 some LH/RH file names are
swapped; the positions stored in each file are used.
"""

from pathlib import Path

from .utils import DATA_ROOT, print_summary, snirf_to_record


def load_inclusion_data(main_dir=DATA_ROOT / "inclusion", fixed_dir=DATA_ROOT / "inclusion_fixed", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*.snirf")):
        fixed = Path(fixed_dir) / path.name
        record = snirf_to_record(fixed if fixed.exists() else path)
        record["subject_id"] = path.parts[-3]  # sub-XXX
        outputs[path.stem] = record

    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_inclusion_data()
