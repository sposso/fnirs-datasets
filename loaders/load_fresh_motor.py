"""
Mesquita, R. C. (2025). FRESH Motor Dataset. OpenNeuro ds005963. doi:10.18112/openneuro.ds005963.v1.0.0

10 participants x 4 sessions of a motor task (sessions "left"/"right" with 25-30 motor events of 2 s
or 3 s; no paper describes it yet): 40 recordings of ~9 min at 8.93 Hz.
68 channels (64 long at 24-39 mm + 4 short at ~8 mm, 760/850 nm) over the sensorimotor cortex of
both hemispheres; all channels are kept.
Optode positions are NIRx's ICBM152 template stored in CapTrak (BIDS coordsystem.json), identical for
every participant; ``head_to_mni`` converts them to MNI, where they coincide with the template points.
"""

from pathlib import Path

from .utils import DATA_ROOT, head_to_mni, print_summary, snirf_to_record


def load_fresh_motor_data(main_dir=DATA_ROOT / "fresh_motor", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*.snirf")):
        record = snirf_to_record(path)
        record["channel_xyz"] = head_to_mni(record["channel_xyz"])  # file positions are CapTrak
        record["subject_id"] = path.parts[-4]  # sub-XX
        outputs[path.stem] = record

    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_fresh_motor_data()
