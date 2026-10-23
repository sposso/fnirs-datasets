"""
Luke, R., Innes-Brown, H., Undurraga, J. A., McAlpine, D. (2022). Human cortical processing of interaural coherence.
iScience 25, 104181. doi:10.1016/j.isci.2022.104181.
Data: OSF tyzag ("Human cortical processing of interaural coherence", BIDS; no license stated); the single zip was
downloaded (SHA256 checked) and extracted without its macOS metadata files. The Mendeley DOI given in the paper
(10.17632/gky2sj9y6x.1) does not resolve.

20 adults listened passively to 5-s band-pass noises (centered at 500 Hz) of different interaural coherence and to
silent control trials, 25 of each in random order, while watching a silent subtitled film; two groups of 10 heard
different intermediate coherence values. One recording of 46-68 min each (18.9 h).
NIRScout (NIRx; 760/850 nm), 16 sources and 16 detectors over the superior temporal gyrus and inferior parietal lobe
of both hemispheres: 88 channels of 24-65 mm, all kept (all "good" in channels.tsv; the paper analyzes only the
20-40 mm ones). Sampled at 7.81 Hz (sub-01 to sub-10) or 3.91 Hz (sub-11 to sub-20).
Optode positions are a template stored in CapTrak (BIDS coordsystem.json), identical for every participant;
``head_to_mni`` converts them to MNI, where they equal the NIRx ICBM152 points of the 10-5 labels listed in the paper.
"""

from pathlib import Path

from .utils import DATA_ROOT, head_to_mni, print_summary, snirf_to_record


def load_interaural_coherence_data(main_dir=DATA_ROOT / "interaural_coherence", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-01/nirs/*.snirf")):
        record = snirf_to_record(path)
        record["channel_xyz"] = head_to_mni(record["channel_xyz"])  # file positions are CapTrak
        record["subject_id"] = path.name.split("_")[0]               # e.g. sub-01
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_interaural_coherence_data()
