"""
Yi, W., Chen, J., Wang, D., Hu, X., Xu, M., Li, F., Wu, S., Qian, J. (2025). A multi-modal dataset of
electroencephalography and functional near-infrared spectroscopy recordings for motor imagery of multi-types of joints
from unilateral upper limb. Scientific Data 12, 953. doi:10.1038/s41597-025-05286-0.
Data: figshare doi:10.6084/m9.figshare.24123303 (FineMI.zip; only the fNIRS folders were extracted).

18 right-handed adults imagined 8 movements of the right hand, wrist, elbow and shoulder (40 trials of 18-20 s per
block, 8 blocks in one session; subject 1's blocks 1-4 are in one folder). One NIRStar folder per block.
NIRScout (760/850 nm, 7.81 Hz), 8 sources and 8 detectors on a 10-5 grid over the left sensorimotor cortex,
24 channels (~32-36 mm); all kept.
Optode positions: NIRx's ICBM152 template stored in CapTrak (NIRS_probeInfo.mat), identical for every participant;
``head_to_mni`` converts them to MNI, where they coincide with the 10-5 points of the paper's Fig. 2.
"""

from pathlib import Path

import mne

from .utils import DATA_ROOT, fill_or_drop, head_to_mni, print_summary, raw_to_record


def load_finemi_data(main_dir=DATA_ROOT / "finemi", verbose=True):
    """Return ``{recording_name: record}`` for every NIRStar block folder in ``main_dir``."""
    outputs = {}
    for folder in sorted(Path(main_dir).glob("subject*/fNIRS/block*")):
        raw = mne.io.read_raw_nirx(folder, verbose="error").load_data(verbose="error")
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw)
        record["channel_xyz"] = head_to_mni(record["channel_xyz"])          # file positions are CapTrak
        subject = folder.parts[-3]                                          # subject1 ... subject18
        record["subject_id"] = subject
        outputs[f"{subject}_{folder.name}"] = record                         # e.g. subject1_block1-4
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_finemi_data()
