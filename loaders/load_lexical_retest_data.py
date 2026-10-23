"""
Elliott, L.M., Rankaduwa, S.S., Hamon-Hill, C., Bode, D.A., Hulls, M.V.M., Lavoie, P.J.A., Newman, A.J. (2026).
Investigating fNIRS test-retest reliability during lexical decision. bioRxiv, doi:10.64898/2026.07.14.737938.
Data: Borealis doi:10.5683/SP4/MXMU9I (BIDS; CC BY-NC-SA 4.0 with additional terms, TERMS.txt); only the raw nirs
files (no accelerometer CSV) and the root files were downloaded, MD5 checked.

30 adults read real words, pseudowords, consonant strings and false fonts (lexical decision) in two sessions one week
apart, 5 runs of ~7.4 min per session (sub-033 has 9 runs). NIRSport2 (NIRx; LED sources, 760/850 nm), 5.09 Hz
(paper: 6.104 Hz), 16 sources and 23 detectors: 42 long channels (29-42 mm) and 8 short channels (7-9 mm) over
the bilateral inferior frontal, temporal and inferior parietal cortices; all channels kept (every channel is "good"
in channels.tsv; the authors' SCI < 0.7 channels are not listed).
Excluded:
- sub-012 and sub-022: no consent to secondary use; the terms require excluding them from any new analysis.
- the 15 runs the authors excluded for more than 8 bad channels (paper; counts in
  derivatives/.../logs/bad_channel_count-all_subjs.csv): all runs of sub-018 and session 2 of sub-030.
Positions: the SNIRF files hold NIRx template positions in CapTrak (coordsystem.json: no digitization); head_to_mni
puts every long-channel optode exactly on a NIRx ICBM152 10-5 point, as for the other CapTrak-template datasets.
The session optodes.tsv files differ from the SNIRF positions (4-40 mm) and are not used.
"""

import csv
from pathlib import Path

import mne

from .utils import DATA_ROOT, fill_or_drop, head_to_mni, print_summary, raw_to_record

NO_CONSENT = {"sub-012", "sub-022"}                     # TERMS.txt / participants.tsv future_use_consent = false
BAD_RUN_LOG = "derivatives/fNIRS_1st_level_0-0.5_glover_derivative/logs/bad_channel_count-all_subjs.csv"
MAX_BAD_CHANNELS = 8                                    # paper: runs with more bad channels were excluded


def excluded_runs(main_dir):
    """``{(subject, session, run)}`` of the runs the authors excluded for too many bad channels."""
    rows = csv.DictReader(open(Path(main_dir) / BAD_RUN_LOG))
    return {(r["Subject"], r["Session"], int(r["Run"])) for r in rows if int(r["N_bad_chans"]) > MAX_BAD_CHANNELS}


def load_lexical_retest_data(main_dir=DATA_ROOT / "lexical_retest", verbose=True):
    """Return ``{recording_name: record}`` for every usable run in ``main_dir``."""
    bad_runs = excluded_runs(main_dir)
    outputs = {}
    for path in sorted(Path(main_dir).glob("sub-*/ses-*/nirs/*_nirs.snirf")):
        subject, session, _, run = path.name.split("_")[:4]                 # sub-006, ses-001, task-ldt, run-01
        if subject in NO_CONSENT or (subject, session, int(run[4:])) in bad_runs:
            continue
        raw = mne.io.read_raw_snirf(path, verbose="error").load_data(verbose="error")
        sfreq = raw.info["sfreq"]
        raw = fill_or_drop(raw, max_gap=round(sfreq), max_edge=round(60 * sfreq))  # 1-s gaps, 1-min start/end
        if raw is None:
            continue
        record = raw_to_record(raw)
        record["channel_xyz"] = head_to_mni(record["channel_xyz"])          # file positions are CapTrak
        record["subject_id"] = subject
        outputs[path.name.removesuffix("_nirs.snirf")] = record
    if verbose:
        print_summary(outputs)
    return outputs


if __name__ == "__main__":
    load_lexical_retest_data()
