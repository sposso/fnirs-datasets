"""Recover the fNIRS signals of the driving recordings (OpenNeuro ds004973) that were uploaded truncated.

25 of the 222 SNIRF files stop before their end, where the HDF5 index and metadata live, so no
software can open them. Their raw signals are stored first: in all 197 intact files the signal block
starts at byte 33,980 and holds 16 float64 channels in the same order, followed (2,368 bytes later)
by the time vector 0, 0.02, 0.04, ... s. We read that block directly and save it as a new SNIRF file,
taking the channel information from an intact recording of the same participant. When the cut falls
inside the signal block, the samples that are present are kept.

Run once from the repository root:  python -m scripts.recover_driving
Writes $FNIRS_DATA_ROOT/driving_recovered/sub-*/nirs/*_nirs.snirf; the originals are not touched.
"""

import h5py
import mne
import numpy as np
from mne_nirs.io import write_raw_snirf

from loaders.utils import DATA_ROOT

SRC = DATA_ROOT / "driving"
DST = DATA_ROOT / "driving_recovered"
START, GAP, N_CHANNELS, SFREQ = 33_980, 2_368, 16, 50.0  # identical in all intact files


def is_readable(path):
    try:
        h5py.File(path, "r").close()
        return True
    except OSError:
        return False


def read_signals(path):
    """Return the raw intensities (samples, 16) stored at the start of a truncated file."""
    data = path.read_bytes()
    time_start = data.find(np.array([0, 1 / SFREQ, 2 / SFREQ], "<f8").tobytes(), START)
    end = time_start - GAP if time_start > 0 else len(data)  # no time vector: file cut inside the signals
    n_samples = (end - START) // (8 * N_CHANNELS)
    return np.frombuffer(data, "<f8", count=n_samples * N_CHANNELS, offset=START).reshape(n_samples, N_CHANNELS)


if __name__ == "__main__":
    for path in sorted(SRC.glob("sub-*/nirs/*_nirs.snirf")):
        if is_readable(path):
            continue
        template = next(p for p in sorted(path.parent.glob("*_nirs.snirf")) if is_readable(p))
        info = mne.io.read_raw_snirf(template, verbose=False).info
        raw = mne.io.RawArray(read_signals(path).T, info, verbose=False)
        out = DST / path.relative_to(SRC)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_raw_snirf(raw, out)
        print(f"{out.name}: {raw.n_times / SFREQ / 60:.1f} min")
