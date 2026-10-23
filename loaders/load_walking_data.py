"""
Minhas, H.S., Nazeer, H., Naseer, N., Khan, U.S., Ansari, A.R., Nawaz, R. (2024). Enhancing classification accuracy
of fNIRS-BCI for gait rehabilitation. IEEE Access 12, 117944-117954. doi:10.1109/ACCESS.2024.3443066.
Data: figshare doi:10.6084/m9.figshare.31871557 ("fNIRS based dataset for motor task", CC BY 4.0; one rar per
participant, MD5 checked).

20 healthy young adults walked on a treadmill: 30-s initial rest, ten trials of 20-s walking and 20-s standing rest,
30-s final rest (one recording of 7.4-7.8 min each). NIRSport2 (NIRx; LED sources, 760/850 nm), 10.17 Hz, 8 sources
and 8 detectors over the left and right motor cortices: 20 channels (37-41 mm as stored; nominal 30 mm in the
paper). The paper excludes no participants or channels; all are kept. Each folder also holds the authors' filtered
SNIRF/CSV files, which are not used.
Optode positions are NIRx's ICBM152 template stored in the files (equal to the NIRx ICBM152 10-5 table: sources at
C1, FC3, CP3, C5, C2, FC4, CP4, C6), identical for every participant; kept as stored, as for the rumor dataset.
The subject numbers in the description.json files do not match the folder names (e.g. sub11 holds "10"), so the
folder name is the subject ID.
"""

from pathlib import Path

from .utils import DATA_ROOT, print_summary, snirf_to_record


def load_walking_data(main_dir=DATA_ROOT / "walking", verbose=True):
    """Return ``{recording_name: record}`` for the raw SNIRF file of every participant folder in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub*/????-??-??_???.snirf")):   # not the filtered *_TLPBW*.snirf
        record = snirf_to_record(path)
        record["subject_id"] = path.parent.name                             # sub1 ... sub20
        outputs[path.parent.name] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_walking_data()
