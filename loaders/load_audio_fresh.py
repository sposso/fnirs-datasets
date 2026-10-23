"""
Luke, R., Shader, M., McAlpine, D. (2025). FRESH Audio Dataset.
OpenNeuro ds005964. doi:10.18112/openneuro.ds005964.v1.0.0

17 participants, one NIRx recording each (20-33 min, 5.21 Hz): passive listening, block design
with speech, noise and control (silence) conditions.
33 channels (25 long + 8 short at ~8 mm, 760/850 nm); all channels are kept.
Optode positions are NIRx's ICBM152 template stored in CapTrak (BIDS coordsystem.json), identical
for every participant; ``head_to_mni`` converts them to MNI, where they coincide with the template
points of the Appearance and Athlete datasets.
"""

from pathlib import Path

from .utils import DATA_ROOT, head_to_mni, print_summary, snirf_to_record


def load_fresh_audio_data(main_dir=DATA_ROOT / "fresh_audio", verbose=True):
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
    load_fresh_audio_data()
