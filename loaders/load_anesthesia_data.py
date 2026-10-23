"""
Saini Ferrón, C., Vargas González, G., Valle Araya, C. (2026). Multimodal EEG-fNIRS data from patients undergoing
general anesthesia. OpenNeuro ds004541 v1.0.0, doi:10.18112/openneuro.ds004541.v1.0.0 (paper in preparation).
Only the fNIRS recordings are used.

8 patients (sub-11 recorded twice), 9 recordings of 20-81 min during induction of and emergence from general
anesthesia (events: baseline, verbal, motor and tetanic stimuli, loss and return of consciousness). NIRx device
(760/850 nm), 7.81 Hz, three montages: 19 channels (sub-02), 20 channels (sub-03 to sub-09) and 15 channels (sub-10,
sub-11), 25-43 mm, all kept. No author flags (every channel "good" in channels.tsv).
Optode positions: template positions stored in CapTrak (coordsystem.json), identical within each montage;
``head_to_mni`` converts them to MNI, where the 15-channel montage coincides with the NIRx ICBM152 10-5 points and all
optodes lie within 4 mm of the fsaverage scalp.
Missing samples: a few zero/negative intensities in sub-11 are handled by utils.fill_or_drop.
"""

from pathlib import Path

import mne

from .utils import DATA_ROOT, fill_or_drop, head_to_mni, print_summary, raw_to_record


def load_anesthesia_data(main_dir=DATA_ROOT / "anesthesia", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*.snirf")):
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw)
        record["channel_xyz"] = head_to_mni(record["channel_xyz"])          # file positions are CapTrak
        record["subject_id"] = path.name.split("_")[0]                      # sub-02 ... sub-11
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_anesthesia_data()
