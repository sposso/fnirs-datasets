"""
Kawabata Duncan, K., Tokuda, T., Sato, C., Tagai, K., Dan, I. (2019). Willingness-to-pay-associated right prefrontal
activation during a single, real use of cosmetics as revealed by functional near-infrared spectroscopy. Frontiers in
Human Neuroscience 13, 16. doi:10.3389/fnhum.2019.00016.
Data: figshare doi:10.6084/m9.figshare.7409834 (v4): one ETG-4000 CSV and one digitizer .pos file per participant.

30 women applied six foundations to their face and rated them (one recording of 10-14 min each). Hitachi ETG-4000
(695/830 nm, 10 Hz), 3x11 holder of 17 illuminators and 16 detectors over the frontal and temporal cortices,
52 channels at 30 mm. The CSV holds raw intensity at both wavelengths; the first 10 s (PreScan, device calibration)
are dropped. Bad channels: the channels marked in each file's "Exception Ch" header line (set on the device) are dropped,
as for the Visuomotor dataset; zero/negative samples are handled by utils.fill_or_drop.
Channels: probes 1-11 (lower row, from the left ear forward and across to the right), 12-22 and 23-33 (upper rows),
alternating illuminators and detectors; channels 1-10 join neighbours of the lower row, 11-21 join the lower and
middle rows, 22-31 the middle row, 32-42 the middle and upper rows, 43-52 the upper row (paper, Fig. 2A).
Positions: the 33 probe positions, nasion and preauricular points digitized for each participant (Polhemus, .pos, mm)
are registered to MNI with utils.coreg_mni (mne.coreg.Coregistration, uniform scaling); Beer–Lambert uses the
digitized source-detector distances.
"""

import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .utils import DATA_ROOT, coreg_mni, fill_or_drop, print_summary, raw_to_record, set_optode_positions

SFREQ = 10.0
WAVELENGTHS = (695, 830)
PRESCAN = 100           # samples of device calibration at the start (PreScan column)
ROW = 11                # probes per row of the 3x11 holder


def channel_pairs():
    """Probe numbers (1-33) joined by channels 1-52, in channel order (paper, Fig. 2A)."""
    pairs = []
    for r in range(3):
        pairs += [(r * ROW + k, r * ROW + k + 1) for k in range(1, ROW)]                # along row r
        if r < 2:
            pairs += [(r * ROW + k, (r + 1) * ROW + k) for k in range(1, ROW + 1)]      # rows r and r + 1
    return pairs


def optode_name(probe):
    """Odd probe numbers are illuminators (17), even ones detectors (16): the holder alternates them."""
    return f"S{(probe + 1) // 2}" if probe % 2 else f"D{probe // 2}"


def read_pos(path):
    """Digitized points of a .pos file, ``{name: xyz in metres}``."""
    text = path.read_text(encoding="latin-1")
    points = {}
    for name, body in re.findall(r"\[([^\]]+)\]\n((?:[A-Z]+=[^\n]*\n?)+)", text):
        values = dict(line.split("=", 1) for line in body.strip().splitlines())
        if all(k in values for k in "XYZ"):
            points[name] = np.array([float(values[k]) for k in "XYZ"]) / 1000.0  # mm -> m
    return points


def read_recording(csv_path, pos_path):
    """Raw intensity of one participant as an MNE Raw without the exception channels (digitized optode positions), and
    the MNI optode positions."""
    lines = csv_path.read_text(encoding="latin-1").splitlines()
    exception = np.array(next(l for l in lines if l.startswith("Exception Ch")).split(",")[1:53], int) == 1
    start = next(k for k, l in enumerate(lines) if l.startswith("Probe1,"))
    data = pd.read_csv(csv_path, skiprows=start, encoding="latin-1")
    intensity = data[[c for c in data.columns if c.startswith("CH")]].to_numpy(float)[PRESCAN:]  # CH1(695), CH1(830), ...
    pairs = channel_pairs()
    keep = np.flatnonzero(~exception)
    names = [f"{optode_name(a)}_{optode_name(b)} {wl}" if a % 2 else f"{optode_name(b)}_{optode_name(a)} {wl}"
             for c in keep for a, b in [pairs[c]] for wl in WAVELENGTHS]
    columns = [2 * c + w for c in keep for w in range(2)]
    info = mne.create_info(names, SFREQ, "fnirs_cw_amplitude")
    for ch in info["chs"]:
        ch["loc"][9] = float(ch["ch_name"].split(" ")[1])                  # wavelength (nm)
    points = read_pos(pos_path)
    probes = np.array([points[f"Probe1-ch{i}"] for i in range(1, 3 * ROW + 1)])
    mni = coreg_mni(probes, points["Nasion"], points["LeftEar"], points["RightEar"])
    set_optode_positions(info, {optode_name(i + 1): xyz for i, xyz in enumerate(probes)})  # Beer–Lambert: real distances
    raw = mne.io.RawArray(intensity[:, columns].T, info, verbose=False)
    return raw, {optode_name(i + 1): xyz for i, xyz in enumerate(mni)}


def load_cosmetic_data(main_dir=DATA_ROOT / "cosmetic", verbose=True):
    """Return ``{recording_name: record}`` for every participant in ``main_dir``."""
    outputs = {}
    for csv_path in sorted(Path(main_dir).glob("Subject*.csv")):
        raw, mni = read_recording(csv_path, csv_path.with_suffix(".pos"))
        raw = fill_or_drop(raw, max_gap=round(SFREQ), max_edge=round(60 * SFREQ))  # 1-s gaps, 1-min start/end
        if raw is None:
            continue
        record = raw_to_record(raw, optode_xyz=mni)
        record["subject_id"] = csv_path.stem                                 # Subject01 ... Subject30
        outputs[csv_path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_cosmetic_data()
