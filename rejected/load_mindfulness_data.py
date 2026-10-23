"""
Gao, Q., Zhang, L. (2023). Brief mindfulness meditation intervention improves attentional control of athletes in
virtual reality shooting competition: Evidence from fNIRS and eye tracking. Psychol. Sport Exerc. 69, 102477.
Data: Mendeley Data doi:10.17632/84gbpd8s5s.1 (one zip of native LUMO recordings, read without extracting it).

University athletes recorded during a baseline and a virtual reality shooting competition, after 20 min of
mindfulness meditation or mind wandering: 142 recordings (folders baseline/shooting x mindfulness/mind-wandering;
the paper reports 76 eligible participants). Folder names (GBK, e.g. "2-10基线" = participant 2-10, baseline) give
the participant. LUMO (Gowerlabs; 735/850 nm), 10 Hz (recordingdata.toml), 9 tiles (12 in 20 recordings).
Native LUMO format read as in Gowerlabs' lumomat (read_lumo.m, load_lumo_enum.m, proc_layout.m): intensity files =
48-byte header + float32 frames of all channels; global source/detector indices = group_location_index in
hardware.toml; sources A/B/C and detectors (ADC channels 0-3, file version 0.1.1) are optodes a/b/c and 1-4 of
layout dock <node_id>. Channels marked inactive by the device (chans_list_act) are dropped; as for the other HD-DOT
arrays, only source-detector pairs up to 35 mm are kept. Zero intensities follow the shared rule (fill_or_drop).
Rejected for the main set: undocumented position frame. layout.json gives 3D optode coordinates (mm) without
landmarks; they are kept as provided.
"""

import json
import re
import tomllib
import zipfile
from pathlib import Path

import mne
import numpy as np

from loaders.utils import DATA_ROOT, fill_or_drop, print_summary, raw_to_record

MAX_DISTANCE = 0.035     # metres (HD-DOT arrays)


def read_lumo(z, folder):
    """Raw intensity of one .LUMO folder inside the zip ``z``, with the layout positions, and its inactive pairs."""
    def toml(name):
        return tomllib.loads(z.read(f"{folder}/{name}").decode())
    variables, hardware = toml("recordingdata.toml")["variables"], toml("hardware.toml")
    chunks = [np.frombuffer(z.read(f"{folder}/{f['file_name']}"), "<f4", offset=48).reshape(-1, variables["n_chans"])
              for f in toml("metadata.toml")["intensity_files"]]
    docks = {int(d["dock_id"].split("_")[1]): {o["optode_id"]: [o["coordinates_3d"][a] for a in "xyz"]
                                                for o in d["optodes"]}
             for d in json.loads(z.read(f"{folder}/layout.json"))["docks"]}
    sources, detectors = {}, {}
    for node in hardware["Hub"]["Group"]["Node"]:
        dock = docks[node["node_id"]]
        for s in node["Source"]:                                      # SRCA_735nm -> optode_a
            sources[s["group_location_index"]] = np.array(dock["optode_" + s["description"][3].lower()]) / 1000
        for d in node["Detector"]:                                    # ADC Detector Channel 0 -> optode_1
            detectors[d["group_location_index"]] = np.array(dock[f"optode_{int(d['description'][-1]) + 1}"]) / 1000
    channels = np.asarray(variables["chans_list"])                   # source, detector, wavelength index
    order = np.lexsort((channels[:, 2], channels[:, 1], channels[:, 0]))  # pairs adjacent for MNE
    wavelengths = variables["wavelength"]
    names = [f"S{s}_D{d} {wavelengths[w - 1]}" for s, d, w in channels[order]]
    info = mne.create_info(names, variables["framerate"], "fnirs_cw_amplitude")
    for ch, (s, d, w) in zip(info["chs"], channels[order]):
        ch["loc"][3:6], ch["loc"][6:9] = sources[s], detectors[d]
        ch["loc"][:3] = (sources[s] + detectors[d]) / 2
        ch["loc"][9] = wavelengths[w - 1]
    active = np.asarray(variables["chans_list_act"]).ravel()[order]
    inactive = {name.split(" ")[0] for name, a in zip(names, active) if not a}
    return mne.io.RawArray(np.concatenate(chunks).T[order], info, verbose=False), inactive


def load_mindfulness_data(main_dir=DATA_ROOT / "mindfulness", verbose=True):
    """Return ``{recording_name: record}`` for every .LUMO folder in the zip file of ``main_dir``."""
    outputs = {}
    with zipfile.ZipFile(next(Path(main_dir).glob("*.zip"))) as z:
        folders = sorted({n.rsplit("/", 1)[0] for n in z.namelist() if n.endswith("recordingdata.toml")})
        for folder in folders:
            raw, inactive = read_lumo(z, folder)
            distances = mne.preprocessing.nirs.source_detector_distances(raw.info)
            far = {name.split(" ")[0] for name, dist in zip(raw.ch_names, distances) if dist > MAX_DISTANCE}
            raw.drop_channels([name for name in raw.ch_names if name.split(" ")[0] in inactive | far])
            sfreq = raw.info["sfreq"]
            raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min edges
            record = raw_to_record(raw)
            name = folder.split("/")[-1].encode("cp437").decode("gbk")     # zip names are stored in GBK
            condition = folder.split("/")[-2]                                # e.g. baseline-mindfulness
            record["subject_id"] = re.match(r"[\d-]+", name).group()       # e.g. 2-10
            record["coord_frame"] = "LUMO layout.json 3D optode coordinates, mm, undocumented frame, no landmarks"
            outputs[f"{condition}_{record['subject_id']}"] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_mindfulness_data()
