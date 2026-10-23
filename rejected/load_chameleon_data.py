"""
Zhao, J., He, X., Jiang, Y., Bao, M. (2026). The Chameleon Paradigm: An effective method for masking biological motion
stimuli. Behav. Res. Methods 58, 59, doi:10.3758/s13428-025-02924-8 (paper and supplement read in full).
Data: Science Data Bank doi:10.57760/sciencedb.25468 (Dataset.zip, manual download; only Study 3 has fNIRS,
data/E3 extracted).

Study 3a (30 adults viewing intact, scrambled and Chameleon biological motion without masks) and Study 3b (30 other
adults viewing masked Chameleon biological motion): one block-design recording of 22-26 min each, 60 in total. File
names are participants' initials: subject_id is anonymous (E3a_sub-01 ... in file order).
LABNIRS (Shimadzu; 780/805/830 nm), recorded at 55.56 Hz and shared downsampled to 18.52 Hz by the authors; bilateral
3 x 3 grids (5 sources, 4 detectors per hemisphere, 30 mm) centered on FC5/FC6: 24 channels, all kept.
The raw files (fNIRSdata/abs/*.TXT) hold absorbance (log10) at three wavelengths: as for the speech comprehension
dataset, 780 and 830 nm are used as optical density (absorbance x ln 10) with MNE's Beer-Lambert law (which takes two
wavelengths); 805 nm and the device HbO/HbR (fNIRSdata/hb) are not used. The optodes of the MNE Raw only encode the
30-mm distance needed by Beer-Lambert.
Rejected for the main set: positions published for 12 of the 24 channels only. Each participant was digitized
(Fastrak) and the group means were converted to MNI with NIRS-KIT, but only significant channels are reported (paper
Tables 6-7, supplement Tables S3-S5); no digitizer files are shared. These MNI values (mm, per study) are kept as
provided in "channel_mni_published" ({channel number: xyz}, channel numbers = the files' channel order);
channel_xyz is None.
"""

from pathlib import Path

import numpy as np

from loaders.utils import DATA_ROOT, intensity_to_raw, print_summary, raw_to_record, read_shimadzu_txt

SD_DISTANCE = 0.03                       # metres (paper)
PUBLISHED_MNI = {                        # group-mean channel MNI (mm) published for each study
    "E3a": {5: [-64.333, -35.667, 45.667], 8: [-62, 4.333, 2.333], 11: [-69, -11.667, -8.333],
            14: [53, 13.333, 46.333], 15: [65.333, -31.667, 48.667], 18: [69, -22.333, 35.667],
            20: [70, -39.667, 24.333]},
    "E3b": {3: [-59, 20.667, 21.667], 15: [66, -35.333, 46.333], 17: [61, 16.667, 26.667],
            18: [70, -27.333, 33.333], 20: [69, -45.333, 21.333], 22: [64, 5.667, 5.333],
            23: [72.667, -37, 5], 24: [71, -11.667, -3.667]},
}


def load_chameleon_data(main_dir=DATA_ROOT / "chameleon", verbose=True):
    """Return ``{recording_name: record}`` for every absorbance file of Study 3 in ``main_dir``."""
    outputs = {}
    for study in ("E3a", "E3b"):
        paths = sorted(Path(main_dir).glob(f"Dataset/data/E3/{study}/fNIRSdata/abs/*.TXT"))
        for number, path in enumerate(paths, start=1):
            pairs, sfreq, absorbance = read_shimadzu_txt(path.read_text(errors="ignore"))  # 780, 805, 830 per channel
            od = np.stack([absorbance[0::3], absorbance[2::3]]) * np.log(10)          # log10 -> natural-log OD
            raw = intensity_to_raw(od, sfreq, (780, 830), SD_DISTANCE, ch_type="fnirs_od")
            record = raw_to_record(raw, allow_2d=True)
            record["channel_xyz"] = None
            record["channel_mni_published"] = PUBLISHED_MNI[study]
            record["subject_id"] = f"{study}_sub-{number:02d}"
            record["coord_frame"] = "none in the files; published group-mean MNI for some channels (mm)"
            outputs[record["subject_id"]] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_chameleon_data()
