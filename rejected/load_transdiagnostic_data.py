"""
Wang, Z., Huang, R., Si, Y., Shen, Z., Peyrodie, L., Nianga, J.-M., Cui, R. (2026). A transdiagnostic prefrontal fNIRS
dataset from 1,228 participants during a verbal fluency task [dataset]. figshare, doi:10.6084/m9.figshare.32022984
(CC BY 4.0); no paper found. Only the raw files (*_MES_Probe1.csv) were extracted from fNIRS_Dataset.zip.

1,228 participants (Health 53, D(epression) 639, GAD 310, SZ 133, BD 93), one verbal fluency task (~2 min) each.
Hitachi ETG, adult 3 x 5 probe (30 mm), 10 Hz, 22 channels; raw light intensity at ~695/830 nm (MNE uses each
channel's measured wavelengths). The groups reuse participant numbers, so each group folder is treated independently.
Author flags: channels marked in the files' "Exception Ch" line (598 channels in 288 files) are dropped. Zero or
negative intensities (103 files) follow the shared missing-data rule (fill_or_drop).
The files start with a UTF-8 byte-order mark, which MNE cannot read: each file is read from a temporary copy without
it (data unchanged). GAD/0757 holds a tab-separated file named 0896 that differs from GAD/0896 (read as GAD_0757,
tabs replaced by commas in the copy). D/0338's MES file holds HbO instead of intensity and is skipped: 1,227
recordings.
Rejected for the main set: no positions (the files name only the probe mode and frontal region; the 2D/3D probe fields
of the shared MATLAB files are empty). The optodes of the MNE Raw only encode the 30-mm distance for Beer-Lambert;
channel_xyz is None.
"""

import tempfile
from pathlib import Path

import mne

from loaders.utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

SD_DISTANCE = 0.03       # metres (Hitachi adult 3 x 5 probe)


def read_mes(text, name):
    """Raw intensity of one MES file (from a copy of its text) and its exception channels."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / name
        copy.write_text(text)
        raw = mne.io.read_raw_hitachi(copy, verbose="error").load_data(verbose="error").pick("fnirs")
    line = next(line for line in text.splitlines() if line.startswith("Exception Ch"))
    flags = [value == "1" for value in line.split(",")[1:]]                  # one per channel CH1 ... CH22
    return raw, {raw.ch_names[2 * i].split(" ")[0] for i, flag in enumerate(flags) if flag}


def load_transdiagnostic_data(main_dir=DATA_ROOT / "transdiagnostic", verbose=True):
    """Return ``{recording_name: record}`` for every MES file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("extracted/Original data(.csv)/*/*/*_MES_Probe1.csv")):
        text = path.read_text(encoding="utf-8-sig").replace("\t", ",")        # GAD/0757: tab-separated
        if "Probe1(Oxy)" in text:                                              # D/0338: HbO only, no intensity
            continue
        raw, flagged = read_mes(text, path.name)
        raw.drop_channels([ch for ch in raw.ch_names if ch.split(" ")[0] in flagged])
        for ch in raw.info["chs"]:
            ch["loc"][3:9] = [0.0, 0.0, 0.0, SD_DISTANCE, 0.0, 0.0]
            ch["loc"][:3] = [SD_DISTANCE / 2, 0.0, 0.0]
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        if raw is None:                                                        # no channel left
            continue
        record = raw_to_record(raw, allow_2d=True)
        record["channel_xyz"] = None
        record["subject_id"] = f"{path.parent.parent.name}_{path.parent.name}"      # e.g. BD_0001
        record["coord_frame"] = "none (3 x 5 probe mode only)"
        outputs[record["subject_id"]] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_transdiagnostic_data()
