"""
Geisler, M., Herbsleb, M., de la Cruz, F., von Au, S., Schumann, A., Croy, I., Bär, K.-J. (2025).
Profound neuronal differences during Exercise-Induced Hypoalgesia between athletes and non-athletes
revealed by functional near-infrared spectroscopy. OpenNeuro ds006902. doi:10.18112/openneuro.ds006902.v1.1.1

42 participants (22 athletes, 20 non-athletes), one NIRSport-2 recording each (~40 min, 7.63 Hz):
pressure pain tests before and after HIIT on a cycle ergometer.
56 channels (48 long + 8 short at ~8 mm, 760/850 nm); all channels are kept.
Optode positions are NIRx's ICBM152 (MNI) template, identical for every participant.
"""

from pathlib import Path

from .utils import DATA_ROOT, print_summary, snirf_to_record


def load_athlete_data(main_dir=DATA_ROOT / "athletes", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*.snirf")):
        record = snirf_to_record(path)
        record["subject_id"] = path.parts[-3]  # sub-VPxx
        outputs[path.stem] = record

    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_athlete_data()
