"""
Duwadi, S., Rogers, D., Boyd, A. D., Carlton, L. B., Zhang, Y., Kawai Gaona, A., Pathiyaparambath, A. D.,
Chaudhury, R., Zimmermann, B. B., O'Brien, W. J., von Lühmann, A., Boas, D. A., Yücel, M. A., Sen, K. (2026).
Decoding spatial attention in the cocktail party problem using wearable whole-head high-density fNIRS. bioRxiv.
doi:10.64898/2026.05.06.722322.
Data: OpenNeuro ds007738 v1.0.0 ("Whole-Head Cocktail Party fNIRS"). The 24 "videoattend" SNIRF files were not
downloaded: they are byte-identical to the "longvisualorient" ones (same recordings, other events).

38 adults; overt and covert audiovisual spatial attention, eye-movement-only control (visualorient), 5-min rest and a
long orienting task (longvisualorient). NINJANIRS22 (Boston University), CW, 760/850 nm, 8.99 Hz, whole head:
56 sources and 144 detectors, 567 pairs from 16 to 42 mm; only pairs up to 35 mm are kept, as for the other
high-density datasets.
Positions: the same AtlasViewer layout for every participant, on AtlasViewer's Colin27 head (voxel frame, axes
left/inferior/anterior), with 329 10-5 landmarks in the same frame. They are turned to RAS axes and moved to MNI
with utils.colin27_optode_mni (landmarks fitted to the same labels of the NIRx ICBM152 table, median error 3.7 mm).
Missing samples: short runs of zero intensity are handled by utils.fill_or_drop.
Skipped: sub-24 visualorient run-01, which has no probe (no positions or wavelengths) and only 795 measurement
descriptions for its 1134 data columns.
"""

from pathlib import Path

import mne

from .utils import DATA_ROOT, colin27_optode_mni, fill_or_drop, print_summary, raw_to_record

MAX_SEPARATION = 0.035  # m, longest source-detector separation kept
SKIPPED = {"sub-24_task-visualorient_run-01_nirs"}  # malformed file: no probe, 795 of 1134 channels described


def load_cocktail_party_data(main_dir=DATA_ROOT / "cocktail_party", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/nirs/*_nirs.snirf")):
        if path.stem in SKIPPED:
            continue
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        optode_xyz = colin27_optode_mni(raw, path)
        distances = mne.preprocessing.nirs.source_detector_distances(raw.info)
        raw.pick([name for name, d in zip(raw.ch_names, distances) if d <= MAX_SEPARATION])
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        record = raw_to_record(raw, optode_xyz=optode_xyz)
        record["subject_id"] = path.name.split("_")[0]                      # e.g. sub-01
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_cocktail_party_data()
