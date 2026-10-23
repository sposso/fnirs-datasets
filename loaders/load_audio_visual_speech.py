"""
Shader, M. J., Luke, R., Gouailhardou, N., McKay, C. M. (2021). The use of broad vs restricted regions of interest in
functional near-infrared spectroscopy for measuring cortical activation to auditory-only and visual-only speech.
Hearing Research 406, 108256. doi:10.1016/j.heares.2021.108256.
Data: "Audio & Visual Speech. Broad vs Restricted Regions of Interest" (OSF xwerv), distributed with MNE-NIRS
(mne_nirs.datasets.audio_or_visual_speech); BIDS.

8 normal-hearing adults; one recording each (31-35 min) of a block design with 18 auditory-only and 18 visual-only
(lipreading) speech segments of a children's story (10-16 s each) and 10 silent control trials.
NIRx NIRScout, 16 LED sources (760/850 nm), 3.91 Hz: 44 long channels (~30 mm) over the left inferior frontal gyrus,
both superior temporal regions and the occipital lobe, plus 8 short channels (8 mm); all are kept.
Optode positions are NIRx's ICBM152 template stored in CapTrak (coordsystem.json), identical for every participant,
as in the FRESH Audio dataset of the same group: after ``head_to_mni`` every source and long-channel detector lies
on the 10-5 position the paper lists for it (0.0 mm).
"""

from pathlib import Path

from .utils import DATA_ROOT, head_to_mni, print_summary, snirf_to_record


def load_audio_visual_speech_data(main_dir=DATA_ROOT / "audio_visual_speech", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*.snirf")):
        record = snirf_to_record(path)
        record["channel_xyz"] = head_to_mni(record["channel_xyz"])  # file positions are CapTrak
        record["subject_id"] = path.parts[-4]                         # sub-XX
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_audio_visual_speech_data()
