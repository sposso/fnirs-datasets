"""
Carlton, L. B., Altınkaynak, M., Kelley, S. M., Zimmermann, B. B., Kura, S., Middell, E., von Lühmann, A.,
Stephen, E. P., Yücel, M. A., Boas, D. A. (2026). Surface-based image reconstruction optimization for high-density
functional near-infrared spectroscopy. Neurophotonics 13(2), 025001. doi:10.1117/1.NPh.13.2.025001.
Data: OpenNeuro ds006673 v1.0.4 ("ball_squeeze_Carlton_2025"), without the raw sourcedata folder.

17 adults: 3 runs of right/left-hand ball squeezing each (51 runs, 4.32 Hz) and one resting-state run (16 runs, none
for sub-577; 14 at 8.99 Hz, 2 at 4.32 Hz). NinjaNIRS (Boston University), CW, 760/850 nm, the same whole-head
probe as the cocktail-party dataset: 56 sources and 144 detectors. The ball-squeezing files and one resting file hold
1220 pairs (up to 60 mm), the other resting files 567; only pairs up to 35 mm are kept (511 and 510 pairs).
Positions: the same AtlasViewer Colin27 template positions and 10-5 landmarks as the cocktail-party dataset,
moved to MNI with utils.colin27_optode_mni.
Missing samples: short runs of zero intensity (at most 4 samples) are handled by utils.fill_or_drop.
"""

from pathlib import Path

import mne

from .utils import DATA_ROOT, colin27_optode_mni, fill_or_drop, print_summary, raw_to_record

MAX_SEPARATION = 0.035  # m, longest source-detector separation kept


def load_ball_squeezing_data(main_dir=DATA_ROOT / "ball_squeezing", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*_nirs.snirf")):
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        optode_xyz = colin27_optode_mni(raw, path)
        distances = mne.preprocessing.nirs.source_detector_distances(raw.info)
        raw.pick([name for name, d in zip(raw.ch_names, distances) if d <= MAX_SEPARATION])
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw, optode_xyz=optode_xyz)
        record["subject_id"] = path.name.split("_")[0]                      # e.g. sub-568
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_ball_squeezing_data()
