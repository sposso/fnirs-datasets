"""Load one downstream dataset and stack its epochs and labels across recordings.

Run from the repository root:  FNIRS_DATA_ROOT=/path/to/data python -m examples.downstream_epochs tinnitus
"""

import sys
from collections import Counter

import numpy as np

from loaders import DOWNSTREAM, load

name = sys.argv[1] if len(sys.argv) > 1 else "tinnitus"
assert name in DOWNSTREAM, f"choose one of {sorted(DOWNSTREAM)}"
records = load(name, verbose=False)

for recording_id, record in list(records.items())[:3]:
    print(recording_id, record["subject_id"], record["epochs"].shape)   # (epochs, channels, [HbO, HbR], samples)

# Montages can differ between participants, so epochs are concatenated per recording, not across them.
labels = np.concatenate([record["labels"] for record in records.values()])
subjects = np.concatenate([[record["subject_id"]] * len(record["labels"]) for record in records.values()])
print(len(labels), "epochs:", dict(Counter(labels)), "from", len(set(subjects)), "participants")
