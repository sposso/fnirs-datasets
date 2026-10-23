"""Shared fNIRS helpers.

``raw_to_record`` turns one recording of raw light intensity (an MNE Raw) into a record:

    raw intensity -> optical density -> Beer–Lambert (HbO/HbR) -> high-pass -> resample

(recordings stored as optical density start at the second step).

``snirf_to_record`` reads a SNIRF file and calls it; datasets in other formats build the Raw themselves
(``read_nirstar`` reads NIRx NIRStar folders that MNE cannot read; ``fill_or_drop`` handles their missing samples).
Channel positions are kept exactly as stored in the file, unless ``optode_xyz`` gives
template positions per optode (datasets without 3D positions or with wrong ones). No cleanup.
``hb_to_record`` does the same from HbO/HbR computed by the device (no raw intensity shared).
``head_to_mni`` converts positions stored in MNE head (CapTrak) coordinates to MNI; ``optode_mni`` fits each
participant's positions and head landmarks (BIDS coordsystem.json) to the fsaverage head and returns them in MNI.
``read_snirf_flat_labels`` reads SNIRF files whose landmark labels MNE cannot read; ``landmark_fit_mni``
moves template positions on another atlas head to MNI with their 10-5 landmarks.
"""

import io
import os
import json
import re
import shutil
import tempfile
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
import scipy.io as sio
from mne.transforms import apply_trans, get_ras_to_neuromag_trans
from scipy.ndimage import label

DATA_ROOT = Path(os.environ.get("FNIRS_DATA_ROOT", "data"))  # one folder per dataset, see README
PPF = 6.0           # partial pathlength factor for Beer–Lambert
L_FREQ = 0.01       # high-pass cutoff (Hz)
TARGET_SFREQ = 7.5  # common sampling rate across datasets (Hz)


def snirf_to_record(snirf_path, verbose=False, **options):
    """Load one SNIRF file and return its record; ``options`` are those of ``raw_to_record``."""
    raw = mne.io.read_raw_snirf(snirf_path, verbose=verbose).load_data(verbose=verbose)
    return raw_to_record(raw, verbose=verbose, **options)


def raw_to_record(raw, ppf=PPF, l_freq=L_FREQ, target_sfreq=TARGET_SFREQ,
                  drop_pairs=(), optode_xyz=None, fix_file_positions=False,
                  channel_pos=None, sd_distance=None, allow_2d=False, verbose=False):
    """Turn raw light intensity (an MNE Raw) into a record with ``data`` shaped (C, 2, T) = [HbO, HbR].

    ``drop_pairs``: source-detector pairs removed before processing, e.g. ``{"S9_D12"}``.
    ``optode_xyz``: ``{optode: xyz in metres}`` that replaces the file's optode positions.
        By default it is applied after Beer–Lambert, so Beer–Lambert and ``sd_distances`` keep
        the file's real source-detector distances and only ``channel_xyz`` uses the template.
        With ``fix_file_positions=True`` it is applied right after reading, because the
        positions stored in the file are wrong.
    ``channel_pos``: ``{"S1_D1": xyz in metres}`` giving each channel's position directly
        (e.g. MNI coordinates published with the dataset); it replaces ``channel_xyz``.
    ``sd_distance``: source-detector distance (metres) used by Beer–Lambert for every channel (or
        ``{"S1_D1": metres}`` per channel), when the real distance is known but the stored positions
        (e.g. a template) do not give it. Each channel keeps its midpoint and direction; only its length changes.
    ``allow_2d``: accept flat 2D positions (rejected datasets, whose positions are kept as provided).
    """
    n_channels_original = len(raw.ch_names)
    raw.drop_channels([ch for ch in raw.ch_names if ch.split(" ")[0] in drop_pairs])
    if fix_file_positions:
        set_optode_positions(raw.info, optode_xyz)
    if sd_distance is not None:
        for ch in raw.info["chs"]:
            length = sd_distance if np.isscalar(sd_distance) else sd_distance[ch["ch_name"].split(" ")[0]]
            direction = (ch["loc"][6:9] - ch["loc"][3:6]) / np.linalg.norm(ch["loc"][6:9] - ch["loc"][3:6])
            ch["loc"][3:6] = ch["loc"][:3] - direction * length / 2
            ch["loc"][6:9] = ch["loc"][:3] + direction * length / 2

    if "fnirs_od" in raw.get_channel_types():                    # the file already holds optical density
        raw_od = raw
    else:
        raw_od = mne.preprocessing.nirs.optical_density(raw)
    haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=ppf)
    highpass_and_resample(haemo, l_freq, target_sfreq, verbose)

    hbo = mne.pick_types(haemo.info, fnirs="hbo")
    hbr = mne.pick_types(haemo.info, fnirs="hbr")
    data = np.stack([haemo.get_data(picks=hbo), haemo.get_data(picks=hbr)], axis=1)
    sd_distances = mne.preprocessing.nirs.source_detector_distances(haemo.info)[hbo]
    if optode_xyz is not None:
        set_optode_positions(haemo.info, optode_xyz)
    channel_xyz = np.array([haemo.info["chs"][p]["loc"][:3] for p in hbo])  # midpoints, metres
    if channel_pos is not None:
        channel_xyz = np.array([channel_pos[haemo.ch_names[p].split(" ")[0]] for p in hbo])

    if not allow_2d and np.all(np.abs(channel_xyz[:, 2]) < 1e-8):
        raise ValueError(f"{raw.filenames[0]}: channel positions are 2D-only")
    if not (np.isfinite(data).all() and np.isfinite(channel_xyz).all()):
        raise ValueError(f"{raw.filenames[0]}: non-finite values in data or channel positions")

    return {
        "data": data,
        "sfreq": haemo.info["sfreq"],
        "n_channels_original": n_channels_original,
        "channel_xyz": channel_xyz,
        "sd_distances": sd_distances,
        "ch_names": [haemo.ch_names[p].split(" ")[0] for p in hbo],           # e.g. "S1_D1"
        "events": list(zip(event_onsets(haemo).tolist(), haemo.annotations.description.tolist())),  # (onset s, name)
    }


def hb_to_record(hbo, hbr, sfreq, channel_xyz, sd_distances, l_freq=L_FREQ,
                 target_sfreq=TARGET_SFREQ, verbose=False):
    """Build a record from HbO and HbR already computed by the device software, shaped (C, T) each, in M.

    For datasets that share no raw intensity. Only the high-pass and resampling of
    ``snirf_to_record`` are applied; ``channel_xyz`` (metres) and ``sd_distances`` (metres) are given
    (``channel_xyz`` is None for rejected datasets that provide no positions).
    """
    n = len(hbo)
    names = [f"CH{i + 1} hbo" for i in range(n)] + [f"CH{i + 1} hbr" for i in range(n)]
    info = mne.create_info(names, sfreq, ["hbo"] * n + ["hbr"] * n)
    haemo = mne.io.RawArray(np.concatenate([hbo, hbr]), info, verbose=verbose)
    highpass_and_resample(haemo, l_freq, target_sfreq, verbose)
    data = haemo.get_data().reshape(2, n, -1).transpose(1, 0, 2)  # (C, 2, T)

    channel_xyz = None if channel_xyz is None else np.asarray(channel_xyz, dtype=float)
    if not (np.isfinite(data).all() and (channel_xyz is None or np.isfinite(channel_xyz).all())):
        raise ValueError("non-finite values in data or channel positions")
    return {
        "data": data,
        "sfreq": haemo.info["sfreq"],
        "n_channels_original": 2 * n,
        "channel_xyz": channel_xyz,
        "sd_distances": np.asarray(sd_distances, dtype=float),
    }


def read_nirstar(folder):
    """Raw intensity of a NIRx NIRStar folder (.hdr, .wl1, .wl2), or of one recording given by its .hdr path,
    as an MNE Raw, without positions.

    For folders MNE cannot read (no probeInfo.mat, NIRStar 14.2). The .hdr gives the sampling rate,
    the wavelengths and which source-detector pairs are channels.
    """
    hdr_path = folder if folder.suffix == ".hdr" else next(
        p for p in folder.glob("*.hdr") if not p.name.startswith("._"))         # skip macOS "._" copies
    hdr = hdr_path.read_text(encoding="latin-1")
    sfreq = float(re.search(r"SamplingRate=([\d.]+)", hdr)[1])                  # samples per second
    wavelengths = re.search(r'Wavelengths="([^"]+)"', hdr)[1].split()            # e.g. ["760", "850"]
    mask_text = re.search(r'S-D-Mask="#\n(.*?)\n#"', hdr, re.S)[1]               # sources x detectors, 1 = channel
    mask = np.array([row.split() for row in mask_text.splitlines()], int)
    sources, detectors = np.nonzero(mask)                                        # the channels (0-based)
    columns = sources * mask.shape[1] + detectors                               # .wl files: a column per source-detector combination
    wl1 = np.loadtxt(hdr_path.with_suffix(".wl1"))[:, columns]                   # first wavelength, time x channels
    wl2 = np.loadtxt(hdr_path.with_suffix(".wl2"))[:, columns]                   # second wavelength

    names = [f"S{s + 1}_D{d + 1} {wl}" for s, d in zip(sources, detectors) for wl in wavelengths]  # e.g. "S1_D1 760"
    info = mne.create_info(names, sfreq, "fnirs_cw_amplitude")
    for ch in info["chs"]:
        ch["loc"][9] = float(ch["ch_name"].split(" ")[1])                       # wavelength (nm)
    x = np.empty((len(names), len(wl1)))                                         # both wavelengths of a channel side by side, as MNE expects
    x[0::2], x[1::2] = wl1.T, wl2.T
    return mne.io.RawArray(x, info, verbose=False)


def fill_or_drop(raw, max_gap, max_edge, positive=True):
    """Handle missing samples (NaN, or zero/negative intensity) of raw light intensity. A channel is a
    source-detector pair (all its wavelengths). With ``positive=False`` (HbO/HbR computed by the device, which can be
    negative) only non-finite samples are missing.
    1. The start and end are cut back to the first and last sample where every channel has data, by at most
       ``max_edge`` samples on each side (a start or end without any data at all is always cut); a channel that
       would need a longer cut is dropped instead.
    2. Inside, gaps of up to ``max_gap`` samples are filled by linear interpolation; a channel with a longer gap is
       dropped, and the cut is redone without it.
    Events (annotations) are kept, moved by the start cut. Returns None if no channel is left."""
    x = raw.get_data()
    x[~np.isfinite(x)] = np.nan
    if positive:
        x[x <= 0] = np.nan                                       # zero or negative intensity is not a measurement
    pairs = {}                                                   # channel ("S1_D1") -> its rows, one per wavelength
    for i, name in enumerate(raw.ch_names):
        pairs.setdefault(name.split(" ")[0], []).append(i)
    rows = list(pairs.values())
    missing = np.array([np.isnan(x[r]).any(axis=0) for r in rows])  # channels x samples, missing at any wavelength
    keep = ~missing.all(axis=1)                                  # channels with some data
    lasting = missing.copy()                                     # missing samples that will not be filled:
    for c in range(len(rows)):                                   # gaps longer than max_gap, or at the start or end
        stretches, n = label(missing[c])
        for s in range(1, n + 1):
            where = np.flatnonzero(stretches == s)
            if len(where) <= max_gap and where[0] > 0 and where[-1] < missing.shape[1] - 1:
                lasting[c, where] = False                        # short gap inside: filled later, ignore for the cut
    while True:
        edges = cut_points(lasting, keep, max_edge)              # may drop channels (updates keep)
        if edges is None:                                        # no channel left
            return None
        first, last = edges
        long_gap = [c for c in np.flatnonzero(keep)              # gap longer than max_gap inside the cut
                    if np.bincount(label(missing[c, first:last + 1])[0])[1:].max(initial=0) > max_gap]
        if not long_gap:
            break
        keep[long_gap] = False                                   # drop them and redo the cut
    x = x[:, first:last + 1]                                     # cut off the start and end
    t = np.arange(x.shape[1])                                    # sample index, for the interpolation
    for c in np.flatnonzero(keep):
        gap = missing[c, first:last + 1]
        for i in rows[c]:
            x[i, gap] = np.interp(t[gap], t[~gap], x[i, ~gap])   # fill the short gaps along a straight line
    kept = {raw.ch_names[i] for c in np.flatnonzero(keep) for i in rows[c]}
    out = mne.io.RawArray(x, raw.info, verbose=False).drop_channels([n for n in raw.ch_names if n not in kept])
    onset = event_onsets(raw) - first / raw.info["sfreq"]       # keep the events, moved by the cut start
    inside = (onset >= 0) & (onset < x.shape[1] / raw.info["sfreq"])
    out.set_annotations(mne.Annotations(onset[inside], raw.annotations.duration[inside],
                                        raw.annotations.description[inside]))
    return out


def event_onsets(raw):
    """Onsets (s) of the events (annotations) of ``raw``, counted from its first sample."""
    start = raw.first_time if raw.annotations.orig_time is not None else 0.0
    return raw.annotations.onset - start


def cut_points(missing, keep, max_edge):
    """First and last sample where every kept channel has data (``missing``: channels x samples), at most
    ``max_edge`` samples in from the first and last sample with any data. Channels that would need a longer cut
    are removed from ``keep`` (in place). Returns None if no channel is left."""
    data = np.flatnonzero(~missing.all(axis=0))                  # samples where some channel has data
    if len(data) == 0:
        return None
    while keep.any():
        good = np.flatnonzero(~missing[keep].any(axis=0))        # samples where every kept channel has data
        if len(good) == 0:                                       # none: drop the channel with the most missing samples
            keep[np.flatnonzero(keep)[missing[keep].sum(axis=1).argmax()]] = False
        elif good[0] - data[0] > max_edge:                       # start cut too long: drop the channels missing there
            keep &= ~missing[:, good[0] - 1]
        elif data[-1] - good[-1] > max_edge:                     # end cut too long: drop the channels missing there
            keep &= ~missing[:, good[-1] + 1]
        else:
            return good[0], good[-1]
    return None


def print_summary(outputs):
    """Print how many recordings a loader returned and their total duration."""
    hours = sum(r["data"].shape[2] / r["sfreq"] for r in outputs.values()) / 3600
    print(f"Loaded {len(outputs)} recordings, {hours:.2f} h")


def highpass_and_resample(haemo, l_freq, target_sfreq, verbose=False):
    """High-pass (4th-order Butterworth, zero-phase) and resample, in place."""
    haemo.filter(l_freq, None, method="iir", phase="zero",
                 iir_params=dict(order=4, ftype="butter"), verbose=verbose)
    haemo.resample(target_sfreq, npad="auto", verbose=verbose)


def set_optode_positions(info, optode_xyz):
    """Set each channel's source, detector and midpoint position from ``{optode: xyz}`` (metres)."""
    for ch in info["chs"]:
        source, detector = ch["ch_name"].split(" ")[0].split("_")  # "S1_D1 760" -> S1, D1
        ch["loc"][3:6] = optode_xyz[source]
        ch["loc"][6:9] = optode_xyz[detector]
        ch["loc"][:3] = (ch["loc"][3:6] + ch["loc"][6:9]) / 2


def head_to_mni(xyz):
    """Convert positions (metres) from MNE head (CapTrak) coordinates to MNI on fsaverage, in metres.

    Uses ``mne.head_to_mni`` with the fsaverage template head and its head<->MRI transform.
    """
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    # head <-> MRI transform of fsaverage
    trans = mne.read_trans(fs_dir / "bem" / "fsaverage-trans.fif")
    return mne.head_to_mni(xyz, "fsaverage", trans, subjects_dir=fs_dir.parent) / 1000.0


def optode_mni(raw, coordsystem):
    """``{optode: MNI xyz in metres}`` from the optode positions in ``raw`` and the landmarks in ``coordsystem``
    (a BIDS coordsystem.json, in mm, in the same frame as the positions).

    The positions are moved to the MNE head frame with the nasion (Nz or NAS) and preauricular points (LPA, RPA), and the
    fsaverage head is fitted to them with ``mne.coreg.Coregistration`` as in MNE's coregistration tutorial, with
    uniform scaling: landmarks first, then ICP with the optodes as head-shape points. The optodes are returned in
    fsaverage's MRI space (MNI305), scaled to the template head."""
    landmarks = {name: np.array(xyz) / 1000.0                              # mm -> m
                 for name, xyz in json.loads(coordsystem.read_text())["AnatomicalLandmarkCoordinates"].items()}
    optodes = {}
    for ch in raw.info["chs"]:
        source, detector = ch["ch_name"].split(" ")[0].split("_")          # "S1_D2 735" -> S1, D2
        optodes[source], optodes[detector] = ch["loc"][3:6], ch["loc"][6:9]
    names = list(optodes)
    nasion = landmarks["Nz"] if "Nz" in landmarks else landmarks["NAS"]    # BIDS files label it either way
    xyz = coreg_mni(np.array([optodes[n] for n in names]), nasion, landmarks["LPA"], landmarks["RPA"])
    return dict(zip(names, xyz))


def coreg_mni(points, nasion, lpa, rpa):
    """MNI positions (metres) of digitized ``points`` (N x 3, metres) given the nasion and preauricular points in the
    same frame: moved to the MNE head frame with the landmarks, then the fsaverage head is fitted to them with
    ``mne.coreg.Coregistration`` (uniform scaling; landmarks first, then ICP with the points as head-shape points)."""
    to_head = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    head = apply_trans(to_head, np.asarray(points))
    info = mne.create_info(1, 1.0, "misc")                                 # only carries the digitization
    info.set_montage(mne.channels.make_dig_montage(
        nasion=apply_trans(to_head, nasion), lpa=apply_trans(to_head, lpa),
        rpa=apply_trans(to_head, rpa), hsp=head, coord_frame="head"))
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    with mne.utils.use_log_level("error"):
        coreg = mne.coreg.Coregistration(info, "fsaverage", fs_dir.parent, fiducials="auto")
        coreg.set_scale_mode("uniform")
        coreg.fit_fiducials()
        coreg.fit_icp(n_iterations=6, nasion_weight=2.0)
        coreg.omit_head_shape_points(distance=5.0 / 1000)
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0)
    return apply_trans(coreg.trans, head) / coreg.scale                    # scaled fsaverage -> fsaverage


def read_snirf_flat_labels(path):
    """Raw intensity of a SNIRF file whose landmark labels are stored as a 1 x N array, which MNE cannot read:
    read from a temporary copy with the labels flattened (data unchanged)."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / Path(path).name
        shutil.copyfile(path, copy)
        with h5py.File(copy, "r+") as f:
            labels = f["nirs/probe/landmarkLabels"][()].ravel()            # (1, N) -> (N,)
            del f["nirs/probe/landmarkLabels"]
            f["nirs/probe/landmarkLabels"] = labels
        return mne.io.read_raw_snirf(copy, verbose="error").load_data(verbose="error")


def intensity_to_raw(intensity, sfreq, wavelengths, sd_distance, ch_type="fnirs_cw_amplitude"):
    """MNE Raw of raw intensity (or optical density, ``ch_type="fnirs_od"``) shaped (2 wavelengths, C, T) for datasets
    that give no optode positions: channel i is S{i+1}_D{i+1}, its source and detector placed ``sd_distance``
    (metres) apart, which only Beer-Lambert uses."""
    n = intensity.shape[1]
    names = [f"S{i + 1}_D{i + 1} {w:g}" for i in range(n) for w in wavelengths]
    info = mne.create_info(names, sfreq, ch_type)
    for ch, name in zip(info["chs"], names):
        ch["loc"][3:9] = [0.0, 0.0, 0.0, sd_distance, 0.0, 0.0]
        ch["loc"][:3] = [sd_distance / 2, 0.0, 0.0]
        ch["loc"][9] = float(name.split(" ")[1])
    data = np.asarray(intensity, float).transpose(1, 0, 2).reshape(2 * n, -1)   # S1 wl1, S1 wl2, S2 wl1, ...
    return mne.io.RawArray(data, info, verbose=False)


def read_shimadzu_txt(text):
    """Channel pairs, sampling rate and data of a Shimadzu text export (LABNIRS, LIGHTNIRS, ...), given its text:
    ``pairs`` [(source, detector)], ``sfreq`` and ``values`` (n_columns x n_samples, 3 columns per channel: oxy,
    deoxy and total Hb, or the absorbance at 3 wavelengths)."""
    lines = text.split("\n", 40)[:40]
    start = int(re.search(r"\[Data Line\]\s+(\d+)", lines[0]).group(1)) - 1        # first data row (0-based)
    pairs = [(int(s), int(d)) for s, d in re.findall(r"\((\d+),(\d+)\)", lines[start - 3])]
    table = pd.read_csv(io.StringIO(text), sep=r"\s+", skiprows=start, header=None, low_memory=False)
    sfreq = 1.0 / np.median(np.diff(table.iloc[:, 0].to_numpy(float)))
    return pairs, sfreq, table.iloc[:, 4:].to_numpy(float).T                        # after Time, Task, Mark, Count


def read_homer_nirs(path):
    """Raw intensity of a Homer ``.nirs`` file (MATLAB d, t, SD) as an MNE Raw, which MNE cannot read itself.
    Optode positions are SD.SrcPos/DetPos as stored, scaled to metres from SD.SpatialUnit."""
    nirs = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    sd = nirs["SD"]
    measurements = np.asarray(sd.MeasList).astype(int)                 # source, detector, -, wavelength index
    order = np.lexsort((measurements[:, 3], measurements[:, 1], measurements[:, 0]))  # pairs adjacent for MNE
    wavelengths = np.asarray(sd.Lambda, float).ravel()
    names = [f"S{s}_D{d} {wavelengths[w - 1]:g}" for s, d, _, w in measurements[order]]
    info = mne.create_info(names, 1.0 / np.median(np.diff(np.ravel(nirs["t"]))), "fnirs_cw_amplitude")
    scale = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[sd.SpatialUnit]
    sources, detectors = np.asarray(sd.SrcPos, float) * scale, np.asarray(sd.DetPos, float) * scale
    for ch, (s, d, _, w) in zip(info["chs"], measurements[order]):
        ch["loc"][3:6], ch["loc"][6:9] = sources[s - 1], detectors[d - 1]
        ch["loc"][:3] = (sources[s - 1] + detectors[d - 1]) / 2
        ch["loc"][9] = wavelengths[w - 1]
    return mne.io.RawArray(np.asarray(nirs["d"], float).T[order], info, verbose=False)


NIRX_ICBM152 = Path(__file__).with_name("nirx_icbm152_1005.json")  # 300 10-5 points (mm), NIRx ICBM152 template


def landmark_fit_mni(points, landmarks):
    """Move ``points`` (N x 3, mm, RAS axes) to MNI in metres, using the 10-5 ``landmarks`` ({label: xyz mm}) given in
    the same frame: ``mne.coreg.fit_matched_points`` (rotation, translation, one scale per axis) maps the landmarks
    onto the same labels of the NIRx ICBM152 table (the positions of the other NIRx-template datasets). For template
    positions on another atlas head (e.g. AtlasViewer's Colin27), where the labels give exact correspondences."""
    target = json.loads(NIRX_ICBM152.read_text())
    common = sorted(set(landmarks) & set(target) - {"Nz", "Iz", "LPA", "RPA"})  # 10-5 points only
    trans = mne.coreg.fit_matched_points(np.array([landmarks[k] for k in common]),
                                         np.array([target[k] for k in common]), scale=3, out="trans")
    if np.linalg.det(trans[:3, :3]) <= 0:
        raise ValueError("landmark fit mirrors the head: check the axes of the positions")
    return apply_trans(trans, np.asarray(points)) / 1000.0


def colin27_to_ras(xyz):
    """Colin27 voxel frame (left, inferior, anterior) to RAS axes, same units."""
    return np.asarray(xyz)[..., [0, 2, 1]] * np.array([-1, 1, -1])


def colin27_optode_mni(raw, path):
    """``{optode: MNI xyz in metres}`` for SNIRF files with AtlasViewer Colin27 template positions (BU NinjaNIRS probe):
    positions and the file's 10-5 landmarks are turned to RAS axes and moved to MNI with ``landmark_fit_mni``."""
    with h5py.File(path, "r") as f:
        labels = [x.decode() for x in f["nirs/probe/landmarkLabels"][()]]
        landmarks = dict(zip(labels, colin27_to_ras(f["nirs/probe/landmarkPos3D"][()][:, :3])))
    optodes = {}
    for ch in raw.info["chs"]:
        source, detector = ch["ch_name"].split(" ")[0].split("_")          # "S1_D2 760" -> S1, D2
        optodes[source], optodes[detector] = ch["loc"][3:6], ch["loc"][6:9]
    names = list(optodes)
    xyz = landmark_fit_mni(colin27_to_ras(np.array([optodes[n] for n in names]) * 1000.0), landmarks)  # m -> mm in
    return dict(zip(names, xyz))
