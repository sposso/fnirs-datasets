"""
Isaev, M. R., Mokienko, O. A., Lyukmanov, R. Kh., Ikonnikova, E. S., Cherkasova, A. N., Suponeva, N. A., Piradov, M. A.,
Bobrov, P. D. (2024). A multiple session dataset of NIRS recordings from stroke patients controlling brain-computer
interface. Scientific Data 11, 1168. doi:10.1038/s41597-024-04012-6. Data: NITRC doi:10.25790/bml0cm.167 (snirf_Raw.zip).

15 stroke patients, 237 motor-imagery BCI sessions of 9 or 14 min (7-24 per patient): imagined left- and right-hand
movements alternating with rest, with online feedback. NIRScout, 760/850 nm, 15.625 Hz: 14 sources and 8 detectors
forming 28 channels (~3 cm) over both motor cortices.
The optodes sit on the 10-10 positions listed in the paper, and the files store exactly their NIRx ICBM152 (MNI)
coordinates, used as stored (as for Appearance). The files are read from the copies made by
scripts/fix_stroke_bci.py, which removes two empty landmark arrays that MNE cannot read.
"""

from pathlib import Path

from .utils import DATA_ROOT, print_summary, snirf_to_record


def load_stroke_bci_data(main_dir=DATA_ROOT / "stroke_bci", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("*.snirf")):
        record = snirf_to_record(path)
        record["subject_id"] = path.stem.split("_")[0]  # e.g. Subject1
        outputs[path.stem] = record                      # e.g. Subject1_day1_session1_raw
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_stroke_bci_data()
