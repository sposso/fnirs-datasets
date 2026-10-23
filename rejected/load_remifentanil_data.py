"""
Karunakaran, K. D., Kussman, B. D., Peng, K., Becerra, L., Labadie, R., Bernier, R., Berry, D., Green, S.,
Zurakowski, D., Alexander, M. E., Borsook, D. (2022). Brain-based measures of nociception during general anesthesia
with remifentanil: A randomized controlled trial. PLoS Med. 19(4), e1003965.
Data: Harvard Dataverse doi:10.7910/DVN/RDDOW1 (CC0); only 10 of its 225 files are public (the rest are restricted).

Public files: two patients (20161027_0728, 20161129_0848) undergoing catheter ablation under general anesthesia,
7 distinct Homer .nirs recordings (six 30-min segments of the intraoperative recordings, one of 5.5 min).
20161129_0848_18-1.nirs is a byte-identical copy of 20161129_0848_18.nirs and the *.orig files are earlier versions:
both are skipped. CW7 (TechEn; 690/830 nm), 25 Hz, 9 sources, 12 long detectors at 30 mm and 9 short detectors at
8 mm: 33 channels, all kept.
Rejected for the main set: no 3D positions. SD.SrcPos/DetPos are a flat 2D layout in mm (z = 0); the paper shows the
probe only on a brain map. Positions are kept as provided.
"""

from pathlib import Path

from loaders.utils import DATA_ROOT, print_summary, raw_to_record, read_homer_nirs


def load_remifentanil_data(main_dir=DATA_ROOT / "remifentanil", verbose=True):
    """Return ``{recording_name: record}`` for every distinct public .nirs file in ``main_dir``."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("*.nirs")):
        if path.stem.endswith("-1"):                                           # identical copy
            continue
        record = raw_to_record(read_homer_nirs(path), allow_2d=True)
        record["subject_id"] = path.stem[:13]                                  # recording date and time = patient
        record["coord_frame"] = "flat 2D layout of the Homer SD (z = 0), mm"
        outputs[path.stem] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_remifentanil_data()
