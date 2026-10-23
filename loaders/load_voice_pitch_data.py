"""
Steinmetzger, K., Megbel, E., Shen, Z., Andermann, M., Rupp, A. (2022). Cortical activity evoked by voice pitch changes:
A combined fNIRS and EEG study. Hearing Research 420, 108483. doi:10.1016/j.heares.2022.108483.
Data: OSF https://osf.io/ps47b ("fNIRS data.zip"; MD5 checked).

20 normal-hearing adults listened passively to 16-s blocks of synthetic vowel sequences with fixed or variable pitch
contours (one session of 61-71 min each, ~21 h). NIRScout 16x16 (NIRx; 760/850 nm, 7.81 Hz; one participant 3.91 Hz):
8 sources and 8 detectors per hemisphere over the auditory cortex, 44 channels (20 of ~30 mm and 2 of ~15 mm per
hemisphere); all kept. Raw intensity from the Homer .nirs files (channel order of ch_config.csv). No channel flags are
shipped (the files contain no NaN; the authors' SCI exclusion was applied only in their analysis).
Positions: the optodes, fiducials and 10-20 points of each participant were digitized with a Polhemus and stored in a
Neuromag FIF file; the point order is documented in the authors' Neuromag2MNI.m (points 1-3 = LPA, Nz, RPA;
points 10-41 = S01, D01, ..., S16, D16). The optodes are registered to MNI with utils.coreg_mni; Beer–Lambert uses
the digitized source-detector distances.
"""

from pathlib import Path

import mne
import numpy as np
import scipy.io as sio

from .utils import DATA_ROOT, coreg_mni, fill_or_drop, print_summary, raw_to_record, set_optode_positions

FIDUCIALS = {"LPA": 0, "Nz": 1, "RPA": 2}   # FIF digitization rows (authors' Neuromag2MNI.m)
OPTODES = range(9, 41)                      # rows of S01, D01, S02, D02, ..., S16, D16


def optode_positions(fif):
    """Digitized and MNI positions, ``({optode: xyz}, {optode: MNI xyz})`` in metres, of one participant."""
    dig = np.array([point["r"] for point in mne.io.read_info(fif, verbose="error")["dig"]])  # head frame, m
    names = [("S" if k % 2 == 0 else "D") + str(k // 2 + 1) for k in range(len(OPTODES))]
    mni = coreg_mni(dig[list(OPTODES)], dig[FIDUCIALS["Nz"]], dig[FIDUCIALS["LPA"]], dig[FIDUCIALS["RPA"]])
    return dict(zip(names, dig[list(OPTODES)])), dict(zip(names, mni))


def read_session(folder):
    """Raw intensity of one session folder (Homer .nirs file) as an MNE Raw with the digitized optode positions,
    and the MNI optode positions."""
    nirs = sio.loadmat(next(folder.glob("*.nirs")), squeeze_me=True, struct_as_record=False)
    sd, time = nirs["SD"], np.ravel(nirs["t"])
    names = [f"S{s}_D{d} {sd.Lambda[w - 1]}" for s, d, _, w in sd.MeasList.astype(int)]
    order = sorted(range(len(names)), key=lambda k: (names[k].split(" ")[0], names[k]))  # wavelengths of a pair together
    info = mne.create_info([names[k] for k in order], 1 / np.median(np.diff(time)), "fnirs_cw_amplitude")
    for ch in info["chs"]:
        ch["loc"][9] = float(ch["ch_name"].split(" ")[1])                  # wavelength (nm)
    digitized, mni = optode_positions(next(folder.glob("*.fif")))
    set_optode_positions(info, digitized)                                  # Beer–Lambert uses the real distances
    return mne.io.RawArray(np.asarray(nirs["d"], float)[:, order].T, info, verbose=False), mni


def load_voice_pitch_data(main_dir=DATA_ROOT / "voice_pitch", verbose=True):
    """Return ``{recording_name: record}`` for every session folder in ``main_dir``."""
    outputs = {}
    for folder in sorted(Path(main_dir).glob("*/*_00*")):
        raw, mni = read_session(folder)
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        if raw is None:
            continue
        record = raw_to_record(raw, optode_xyz=mni)
        record["subject_id"] = folder.name                                  # one session per participant
        outputs[folder.name] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_voice_pitch_data()
