"""
Medvedeva, A., Syrov, N., Yakovlev, L., Alieva, Y., Berkmush-Antipova, A., Ivanova, G., Shusharina, N., Kaplan, A.
(2026). Multisession fNIRS-EEG data of post-stroke motor recovery. Recordings during intact and paretic hand
movements. Scientific Data 13, 448. doi:10.1038/s41597-026-06803-5. Data: figshare doi:10.6084/m9.figshare.28904942
(FNIRS_SNIRF.zip).

16 patients with upper-limb hemiparesis, 3-6 sessions each (84 sessions) during a two-week rehabilitation program.
In each run of a stimulus-response task (12 LED flashes, one every 9-11 s) the patient pressed a button with the
target hand (paretic or intact) when its LED flashed: 475 runs of 2-16 min.
NIRSport 16x16 / 8x8, 760/850 nm, 13.56 Hz: 24 sources and 22 working detectors forming 70 channels (~3 cm) over the
sensorimotor and adjacent areas; the authors excluded a malfunctioning detector and its symmetric counterpart.
The files store optical density and the NIRx ICBM152 (MNI) coordinates of the optodes' 10-20 positions (paper, Fig. 2),
used as stored. They are read from the copies made by scripts/fix_stroke_recovery.py, which transposes the data,
labels them as optical density and corrects the positions of three detectors (P4, P3, Pz).
Events: the SNIRF stim group ("32") is a placeholder, identical in every file; the real events are the authors'
event tables (snirf_data/events/<file>_evt.txt: sample, 0, code), read with mne.read_events. Code 1 = left LED flash
("flash_l"), 2 = right flash ("flash_r") (README). The file name gives the run's target hand (e.g. ..._left1): a
flash on that side is a button press, the other side is a non-target without movement. Downstream task (GeoNIRS):
movement vs no movement, one sample per flash.
"""

from pathlib import Path

import mne

from ..utils import DATA_ROOT, print_summary, raw_to_record
from . import epochs

CONDITION = (0.0, 12.0)  # start after the event and duration (s); paper: 200-ms flash every 9-11 s, epoched by the authors' script (Preprocessing.ipynb: 12 s)

EVENTS_DIR = DATA_ROOT / "stroke_recovery_download/snirf_data/events"  # the authors' *_evt.txt tables


def load_stroke_recovery_data(main_dir=DATA_ROOT / "stroke_recovery", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("*.snirf")):
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        events = mne.read_events(EVENTS_DIR / f"{path.stem}_evt.txt")
        raw.set_annotations(mne.annotations_from_events(events, raw.info["sfreq"], {1: "flash_l", 2: "flash_r"}))
        record = raw_to_record(raw)
        record["subject_id"] = path.stem[:4]  # e.g. L_01 (paretic side and patient number)
        hand = path.stem.split("_")[-1].rstrip("0123456789")                # target hand of the run: left or right
        record["event_labels"] = {"flash_l": "movement" if hand == "left" else "no movement",
                                  "flash_r": "movement" if hand == "right" else "no movement"}
        record["epochs"], record["labels"] = epochs(record, *CONDITION)
        outputs[path.stem] = record           # e.g. L_01_session2_left1
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_stroke_recovery_data()
