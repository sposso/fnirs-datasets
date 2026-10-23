"""
Ge, Q., et al. (2024). The effect of empathy on emotional response to Pipa music: evidence from fNIRS [in Chinese].
ChinaXiv:202411.00062 (preprint submitted to Acta Psychologica Sinica).
Data: Science Data Bank doi:10.57760/sciencedb.14924 (V1, CC BY 4.0, "Effects of empathy on emotional responses to
Pipa music (behavioural and fNIRS data)"); only the raw NIRStar files were downloaded (not the nirsSPM results).

High- and low-empathy university students listened to 14 Pipa excerpts (7 positive, 7 negative), each rated on
valence, arousal and tension and on 6 basic and 9 aesthetic emotions: one recording of ~40 min per participant.
The paper analyzed 49 participants; the dataset holds 45 recordings (folders 1-46 without 20, whose folder has no
light-intensity files). Folders 44-46 are stored twice (top level and n=46/), byte-identical; they are read once.
NIRScoutX (NIRx; LED sources, 760/850 nm), 24 sources and 24 detectors on standard 10-5 positions: 81 channels
(20-44 mm) over the prefrontal, temporal and parietal cortex; 2.60 Hz (the device rate shared by 24 time-multiplexed
sources; the paper quotes the 7.81 Hz device maximum). All channels are kept.
Optode positions are the NIRx ICBM152 template of the probe (probeInfo.mat; digpts.txt equals the NIRx ICBM152 table),
identical for every participant: the paper's digitized positions are not shared. MNE stores them in its head frame;
``head_to_mni`` converts them back to the template MNI positions exactly, as for FRESH Audio.
"""

import tempfile
from pathlib import Path

import mne

from .utils import DATA_ROOT, head_to_mni, print_summary, raw_to_record


def read_nirstar_folder(folder):
    """Raw intensity of one NIRStar folder. Three .evt files (folders 14, 30, 34) end with an empty line, which MNE
    cannot parse: those folders are read from a temporary copy (links to the files) with the empty lines removed."""
    evt = next(folder.glob("*.evt"))
    lines = evt.read_text().splitlines()
    if all(line.strip() for line in lines):
        return mne.io.read_raw_nirx(folder, verbose="error").load_data(verbose="error")
    with tempfile.TemporaryDirectory() as tmp:
        for f in folder.iterdir():
            if f.is_file() and f != evt:
                (Path(tmp) / f.name).symlink_to(f.resolve())
        (Path(tmp) / evt.name).write_text("\n".join(line for line in lines if line.strip()) + "\n")
        return mne.io.read_raw_nirx(tmp, verbose="error").load_data(verbose="error")


def load_pipa_music_data(main_dir=DATA_ROOT / "pipa_music", verbose=True):
    """Return ``{recording_name: record}`` for every NIRStar folder with light-intensity files in ``main_dir``."""
    folders = {}                                                            # participant number -> folder
    for wl1 in sorted(Path(main_dir).glob("**/*.wl1")):
        folder = wl1.parent
        if folder.name.isdigit():
            folders.setdefault(int(folder.name), folder)                    # 44-46 also under n=46/: keep one
    outputs = {}
    for number, folder in sorted(folders.items()):
        raw = read_nirstar_folder(folder)
        record = raw_to_record(raw)
        record["channel_xyz"] = head_to_mni(record["channel_xyz"])          # MNE head frame -> template MNI
        record["subject_id"] = f"sub-{number:02d}"
        outputs[f"sub-{number:02d}_{next(folder.glob('*.wl1')).stem}"] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_pipa_music_data()
