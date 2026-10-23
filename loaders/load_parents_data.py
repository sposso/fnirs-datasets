"""
Bizzego, A., Gabrieli, G., Azhari, A., Lim, M., Esposito, G. (2022). Dataset of parent-child hyperscanning
functional near-infrared spectroscopy recordings. Scientific Data 9, 625. doi:10.1038/s41597-022-01751-2.
Data: DR-NTU (Data Repository of the Nanyang Technological University), CC BY-NC.

62 parent-child dyads (33 mother-child, 29 father-child; children aged 3-5 years), each person recorded
during a video session (~3.6 min) and a free-play session (~10.5 min). NIRx NIRSport, 7.81 Hz,
760/850 nm, 8 sources and 7 detectors forming 20 prefrontal channels; the same montage on adult and
child caps.
scripts/convert_parents.py converts the NIRStar files to SNIRF with utils.fill_or_drop: a missing start or end of
up to 1 min is cut off, gaps of up to 1 s are filled by linear interpolation, channels with longer gaps are dropped,
and every recording is kept with the channels it has left.
The files hold the positions of NIRx's standard Prefrontal 8x8 montage: each optode on a 10-20 position,
at its NIRx ICBM152 (MNI) coordinates. Beer–Lambert uses the 30-mm source-detector distance listed in the
NIRStar headers for every channel.
"""

from pathlib import Path

from .utils import DATA_ROOT, print_summary, snirf_to_record

SD_DISTANCE = 0.030  # metres: the distance of every channel in the NIRStar headers (ChanDis)


def load_parents_data(main_dir=DATA_ROOT / "parents", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("*/*.snirf")):                  # video/ and play/ folders
        record = snirf_to_record(path, sd_distance=SD_DISTANCE)        # intensity -> HbO/HbR at 7.5 Hz
        record["subject_id"] = "_".join(path.stem.split("_")[:2])          # "NSFD1_child_video" -> "NSFD1_child"
        outputs[path.stem] = record                                        # key: "<dyad>_<role>_<session>"

    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_parents_data()
