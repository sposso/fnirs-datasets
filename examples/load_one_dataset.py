"""Load one pretraining dataset and print the structure of its first record.

Run from the repository root:  FNIRS_DATA_ROOT=/path/to/data python -m examples.load_one_dataset fresh_motor
"""

import sys

from loaders import load

name = sys.argv[1] if len(sys.argv) > 1 else "fresh_motor"
records = load(name, verbose=True)                  # {recording_id: record}
recording_id, record = next(iter(records.items()))

print(recording_id, record["subject_id"])
print("data (channels, [HbO, HbR], samples):", record["data"].shape, "at", record["sfreq"], "Hz")
print("channel_xyz (MNI, m):", record["channel_xyz"].shape)
print("first channels:", record["ch_names"][:3], "SD distances (m):", record["sd_distances"][:3].round(4))
print("first events:", record["events"][:3])
