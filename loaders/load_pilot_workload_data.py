"""
Hamann, A., Kneffel, R. J., Cyrol, D., Rankova, E., Sammito, S., Carstengerdes, N. (2026). Examining physiological
parameters of mental workload in the cockpit: a multimethod approach. Frontiers in Physiology 17, 1893570.
doi:10.3389/fphys.2026.1893570.
Data: OSF doi:10.17605/OSF.IO/YXRKE ("Multimethod data on mental workload in pilots", CC BY 4.0); only
fNIRS/fNIRS_raw.zip (SHA256 checked), its readme and the exclusion list were downloaded.

14 commercial airline pilots flew a simplified cruise flight in an A320 simulator while doing an auditory n-back
heading task (0- to 3-back) and an altitude-monitoring task, in 8 blocks of about 5 min; one recording of 65-78 min
each (16.9 h). 16 pilots were recruited; the 2 excluded for sleepiness (Vpn12, Vpn15) are not in the shared files.
NIRSport2 (NIRx; 760/850 nm), 10.17 Hz, 8 sources and 7 detectors forming 17 channels of 26-39 mm over the bilateral
prefrontal cortex, plus 8 short-separation channels (7-8 mm, one per source); all kept. Same montage as the
simulated-flight dataset (load_flight_data).
Optode positions are NIRx's ICBM152 template stored in the files (S1-S8 and D1-D7 equal the NIRx ICBM152 points of
the montage's 10-5 labels), i.e. MNI coordinates, identical for every participant; kept as stored.
"""

from pathlib import Path

from .utils import DATA_ROOT, print_summary, snirf_to_record


def load_pilot_workload_data(main_dir=DATA_ROOT / "pilot_workload", verbose=True):
    """Return ``{recording_name: record}`` for every SNIRF file in ``main_dir`` (one folder per participant)."""
    outputs = {}
    for path in sorted(Path(main_dir).glob("Vpn*/*.snirf")):
        record = snirf_to_record(path)
        record["subject_id"] = path.parent.name                    # e.g. Vpn01
        outputs[path.parent.name] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_pilot_workload_data()
