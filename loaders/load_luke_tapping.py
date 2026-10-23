"""
Luke, R., McAlpine, D. (2021). fNIRS Finger Tapping Data in BIDS Format (v0.0.1). Zenodo. doi:10.5281/zenodo.5529797
(also on GitHub as rob-luke/BIDS-NIRS-Tapping; no paper describes it).

5 adults, one recording of 40-50 min each: tapping the left or right thumb to the fingers for 5 s, or a control
condition, cued by sound. NIRx device (760/850 nm), 7.81 Hz, 28 channels (20 long up to 41 mm + 8 short at ~7 mm)
over the motor cortex of both hemispheres; all channels are kept (all are "good" in channels.tsv).
Optode positions are a template stored in CapTrak (BIDS coordsystem.json), identical for every participant;
``head_to_mni`` converts them to MNI, as for FRESH Motor.
"""

from pathlib import Path

from .utils import DATA_ROOT, head_to_mni, print_summary, snirf_to_record


def load_luke_tapping_data(main_dir=DATA_ROOT / "luke_tapping", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*.snirf")):
        record = snirf_to_record(path)
        record["channel_xyz"] = head_to_mni(record["channel_xyz"])  # file positions are CapTrak
        record["subject_id"] = path.name.split("_")[0]               # e.g. sub-01
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_luke_tapping_data()
