"""
Li, X., Krol, M. A., Jahani, S., Boas, D. A., Tager-Flusberg, H., Yücel, M. A. (2020). Brain correlates
of motor complexity during observed and executed actions. Scientific Reports 10, 10965.
doi:10.1038/s41598-020-67327-5. OpenNeuro ds005935.

21 participants x 3 runs (~6 min) of observing and executing simple and complex right-hand actions,
plus a visual control. TechEn CW7, 690/830 nm, 25 Hz.
60 channels over both hemispheres, from the inferior frontal to the posterior parietal cortex:
52 long (30 mm) and 8 short (8 mm); all kept.
The files hold only a flat 2D layout, which keeps the real distances used by Beer–Lambert. Each channel
gets the MNI coordinates of the paper's supplementary Table S2, used as given: AtlasViewer projections
on the cortex, numbered in the files' measurement order. The left-hemisphere points lie ~12 mm closer
to the midline than their right-hemisphere mirrors.
Events: the BIDS events.tsv (5-s trials of EasyObservation, DifficultObservation, EasyExecution, DifficultExecution
and Control, 6-13 s apart). Downstream task (GeoNIRS): observation vs execution (simple and complex pooled; Control
not used), one sample per trial.
"""

from pathlib import Path

import mne
import numpy as np

from ..utils import DATA_ROOT, print_summary, raw_to_record
from . import bids_events, epochs

# Not loaded: a trimmed copy of another run, and two runs with corrupted values (e.g. 1e245)
SKIP = {
    "sub-6110_ses-01_task-MirrorNeuron_run-3_nirs",  # copy of run-2
    "sub-6110_ses-01_task-MirrorNeuron_run-2_nirs",  # S7_D12 at 690 nm corrupted between 20 and 51 s
    "sub-6112_ses-01_task-MirrorNeuron_run-2_nirs",  # S16_D29 and S16_D30 at 830 nm corrupted throughout
}

# Channel MNI coordinates in mm (paper, supplementary Table S2: channels 1-60 in the files' order)
CHANNEL_MNI = {
    # Right hemisphere (S1-S8)
    "S1_D1": (44, -67, 13), "S1_D2": (63, -64, 18), "S1_D7": (44, -62, 21),
    "S1_D10": (51, -76, 28), "S1_D11": (39, -50, 26), "S2_D2": (68, -51, 19),
    "S2_D3": (56, -41, 17), "S2_D11": (60, -50, 29), "S2_D12": (48, -42, 26),
    "S3_D3": (64, -23, 18), "S3_D4": (73, -12, 16), "S3_D8": (62, -15, 24),
    "S3_D12": (52, -21, 30), "S3_D13": (56, -11, 30), "S4_D4": (70, 2, 15),
    "S4_D5": (57, 10, 13), "S4_D13": (60, 1, 30), "S4_D14": (64, 13, 28),
    "S5_D5": (63, 21, 12), "S5_D6": (63, 40, 12), "S5_D9": (48, 22, 17),
    "S5_D14": (58, 22, 26), "S5_D15": (62, 39, 24), "S6_D11": (58, -50, 43),
    "S6_D12": (45, -35, 41), "S7_D12": (61, -24, 47), "S7_D13": (59, -12, 45),
    "S7_D16": (45, -17, 44), "S8_D13": (53, 0, 40), "S8_D14": (55, 13, 36),
    # Left hemisphere (S9-S16)
    "S9_D17": (-30, 30, 13), "S9_D18": (-56, 19, 12), "S9_D23": (-35, 16, 22),
    "S9_D26": (-53, 36, 20), "S9_D27": (-47, 21, 24), "S10_D18": (-41, 6, 13),
    "S10_D19": (-56, -1, 10), "S10_D27": (-52, 10, 24), "S10_D28": (-61, 4, 25),
    "S11_D19": (-63, -16, 14), "S11_D20": (-41, -24, 12), "S11_D24": (-49, -20, 18),
    "S11_D28": (-46, -17, 24), "S11_D29": (-51, -25, 27), "S12_D20": (-47, -43, 14),
    "S12_D21": (-40, -45, 14), "S12_D29": (-43, -44, 25), "S12_D30": (-56, -55, 29),
    "S13_D21": (-37, -58, 14), "S13_D22": (-37, -66, 11), "S13_D25": (-42, -71, 23),
    "S13_D30": (-40, -61, 25), "S13_D31": (-31, -66, 28), "S14_D27": (-40, 11, 28),
    "S14_D28": (-38, 3, 28), "S15_D28": (-38, -16, 36), "S15_D29": (-44, -23, 37),
    "S15_D32": (-47, -23, 42), "S16_D29": (-41, -40, 36), "S16_D30": (-51, -54, 39),
}
CHANNEL_POS = {name: np.array(xyz) / 1000.0 for name, xyz in CHANNEL_MNI.items()}  # mm -> m
TASK_LABELS = {"EasyObservation": "observation", "DifficultObservation": "observation",
               "EasyExecution": "execution", "DifficultExecution": "execution"}


CONDITION = (0.0, 4.0)  # start after the event and duration (s); paper: "each trial lasted 4 s" (events.tsv says 5 s)

def load_mirror_data(main_dir=DATA_ROOT / "mirror", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*.snirf")):
        if path.stem in SKIP:
            continue
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        raw.set_annotations(bids_events(path))                              # names instead of the stim codes
        record = raw_to_record(raw, channel_pos=CHANNEL_POS)
        record["subject_id"] = path.parts[-4]  # sub-XXXX
        record["event_labels"] = TASK_LABELS
        record["epochs"], record["labels"] = epochs(record, *CONDITION)
        outputs[path.stem] = record

    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_mirror_data()
