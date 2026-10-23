"""
Feng, L., Xu, B., Zhang, H., Lin, B., Deng, Z., Tao, S., Liu, C., Jia, S., Duan, L., Jia, Z. (2026). A Multimodal
fNIRS-EEG Dataset for Unilateral Limb Motor Imagery. Scientific Data 13, 1159. doi:10.1038/s41597-026-07807-x.
Data (MIND): ScienceDB doi:10.57760/sciencedb.34326, version V4 (only the fNIRS files of MIND_BIDS were downloaded).

30 adults imagining moving the right arm in four directions (left to right, top to bottom and the two diagonals), in
3 runs of ~23 min, each with two sessions of 20 trials and rest periods: 90 recordings (sub-07's second run was
interrupted and is in two parts; sub-13's second run was corrupted). Shimadzu LABNIRS (780/805/830 nm) at 47.62 Hz:
18 sources and 18 detectors forming 51 channels (30 mm) over the sensorimotor and parietal cortex of both hemispheres.
The SNIRF files hold the HbO and HbR computed by the Shimadzu software (identical to the vendor export in sourcedata/),
as concentration x path length (mM*cm, as in REFED), divided by 18 cm (30 mm x partial pathlength factor 6) to get M.
Their channel labels are wrong: the data are the HbO of the 51 channels followed by their HbR, in the channel order of
channels.tsv (the vendor's order). Channel 25 is listed as S8-D10 (71 mm), but the vendor export gives S10-D8 (38 mm,
the mirror of S5-D3), which is used. Two recordings have two channels without data (all zero), which are dropped.
Each optode lies exactly midway between two electrodes of the authors' 10-20 cap layout (sub-*_electrodes.tsv), i.e. on
a 10-5 position, and gets the MNI position of that position on the NIRx ICBM152 template.
Events (rest, MI start, the four MI directions) are read from the SNIRF stim groups, present in every file.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .utils import DATA_ROOT, PPF, hb_to_record, print_summary

SD_DISTANCE = 0.030                            # m, every channel (paper)
TO_MOLAR = 1e-3 / (SD_DISTANCE * 100 * PPF)    # mM*cm -> M, dividing by 18 cm
FIXED_PAIRS = {("S8", "D10"): ("S10", "D8")}   # channel 25, as in the vendor export

# optode: (10-5 position, midway between two electrodes of the authors' cap; NIRx ICBM152 MNI position in mm)
OPTODES = {
    # Sources
    "S1": ("FFT7", (-75.968, 28.588, -19.004)), "S2": ("FFC3", (-58.244, 40.554, 45.750)),
    "S3": ("FCC5", (-81.276, 2.703, 22.051)), "S4": ("TTP7", (-87.623, -31.634, -15.332)),
    "S5": ("CCP3", (-72.036, -29.818, 65.923)), "S6": ("CPP5", (-77.543, -63.508, 27.478)),
    "S7": ("FFC4", (57.431, 41.450, 46.201)), "S8": ("FFT8", (75.554, 29.800, -18.205)),
    "S9": ("FCC6", (80.908, 4.406, 22.797)), "S10": ("CCP4", (71.538, -28.329, 66.603)),
    "S11": ("TTP8", (86.666, -29.825, -14.383)), "S12": ("CPP6", (77.912, -62.369, 28.299)),
    "S13": ("FFC1", (-33.046, 44.916, 68.640)), "S14": ("FFC2", (32.190, 45.412, 68.838)),
    "S15": ("FCCz", (-0.421, 10.765, 95.390)), "S16": ("CCP1", (-40.486, -28.942, 92.993)),
    "S17": ("CCP2", (39.600, -28.111, 93.392)), "S18": ("CPPz", (-0.455, -67.110, 96.112)),
    # Detectors
    "D1": ("FFC5", (-72.304, 34.818, 14.923)), "D2": ("FTT7", (-82.269, -1.198, -17.312)),
    "D3": ("FCC3", (-67.280, 6.436, 58.396)), "D4": ("CCP5", (-84.960, -30.752, 26.592)),
    "D5": ("TPP7", (-79.389, -61.215, -10.304)), "D6": ("CPP3", (-64.664, -65.552, 62.557)),
    "D7": ("FFC6", (71.703, 35.942, 15.530)), "D8": ("FCC4", (66.383, 7.825, 58.862)),
    "D9": ("FTT8", (81.952, 0.528, -16.537)), "D10": ("CCP6", (84.942, -28.987, 27.557)),
    "D11": ("CPP4", (64.597, -64.616, 63.414)), "D12": ("TPP8", (79.798, -60.035, -9.687)),
    "D13": ("FFCz", (-0.312, 46.899, 78.222)), "D14": ("FCC1", (-38.088, 9.315, 84.553)),
    "D15": ("FCC2", (37.198, 10.092, 84.755)), "D16": ("CCPz", (-0.483, -28.307, 103.974)),
    "D17": ("CPP1", (-36.905, -66.856, 87.375)), "D18": ("CPP2", (36.235, -66.333, 87.974)),
}
OPTODE_XYZ = {name: np.array(xyz) / 1000.0 for name, (_, xyz) in OPTODES.items()}  # mm -> m


def read_channels(path):
    """Source and detector of the 51 channels, in the order of the data (channels.tsv has a row per wavelength)."""
    table = pd.read_csv(path, sep="\t")
    table = table[table.wavelength_nominal == 780]                       # one row per channel
    return [FIXED_PAIRS.get((s, d), (s, d)) for s, d in zip(table.source, table.detector)]


def read_stim(nirs):
    """(onset s, name) of every stimulus in the SNIRF ``stim*`` groups, in time order (same content as the
    BIDS events.tsv, which is missing for sub-17 to sub-30)."""
    events = []
    for key in (k for k in nirs if k.startswith("stim")):
        name = np.asarray(nirs[key]["name"][()]).ravel()[0]
        name = name.decode() if isinstance(name, bytes) else str(name)
        events += [(float(row[0]), name) for row in np.atleast_2d(nirs[key]["data"][()])]
    return sorted(events)


def load_mind_data(main_dir=DATA_ROOT / "mind", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*_nirs.snirf")):
        pairs = read_channels(path.with_name(path.name.replace("_nirs.snirf", "_channels.tsv")))
        with h5py.File(path, "r") as f:
            data = f["nirs/data1/dataTimeSeries"][()] * TO_MOLAR          # time x 102: HbO of the 51 channels, then HbR
            time = f["nirs/data1/time"][()]
            events = read_stim(f["nirs"])
        hbo, hbr = data[:, :51].T, data[:, 51:].T
        keep = np.flatnonzero(hbo.any(axis=1) | hbr.any(axis=1))         # drop channels without data (all zero)
        channel_xyz = np.array([(OPTODE_XYZ[s] + OPTODE_XYZ[d]) / 2 for s, d in pairs])  # midpoints
        record = hb_to_record(hbo[keep], hbr[keep], 1 / np.median(np.diff(time)), channel_xyz[keep],
                              np.full(len(keep), SD_DISTANCE),
                              ch_names=[f"{s}_{d}" for s, d in np.array(pairs)[keep]],
                              events=[(onset - time[0], name) for onset, name in events])
        record["subject_id"] = path.parts[-3]                            # sub-XX
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_mind_data()
