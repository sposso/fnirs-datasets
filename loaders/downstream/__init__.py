"""Downstream-task datasets, held out from pretraining: the six GeoNIRS downstream datasets we have.

The loaders return records like the pretraining loaders, plus:
- ``events``: every event of the recording, ``(onset s, name)`` on the record's time axis;
- ``event_labels``: ``{event name: label}`` for the events of the task (GeoNIRS, Supplementary Table 1 and
  github.com/GeoNIRS/GeoNIRS/tree/main/benchmark/downstream);
- ``group`` for the tasks labelled by participant group (ICH, CI status, tinnitus);
- ``epochs`` (n, C, 2, T) and ``labels``: one epoch per labelled event, covering the whole condition as the
  dataset's paper describes it (``CONDITION`` in each loader: start after the event and duration, in s).
"""

import mne
import mne_bids
import numpy as np


def bids_events(snirf_path):
    """Events of the BIDS ``events.tsv`` next to ``snirf_path``, as MNE annotations (read with MNE-BIDS)."""
    kwargs = mne_bids.events_file_to_annotation_kwargs(str(snirf_path).replace("_nirs.snirf", "_events.tsv"),
                                                       verbose=False)
    return mne.Annotations(kwargs["onset"], kwargs["duration"], kwargs["description"])


def epochs(record, start, duration):
    """One epoch per labelled event of ``record``, from ``start`` s after the event for ``duration`` s, with MNE
    (annotations -> events_from_annotations -> Epochs; no baseline, detrending or rejection). Epochs that leave
    the recording are dropped. Returns X (n_epochs, C, 2, T) = [HbO, HbR] and the label of each epoch."""
    tmin, tmax = start, start + duration - 1 / record["sfreq"]             # MNE includes the sample at tmax
    n = len(record["ch_names"])
    info = mne.create_info([f"{c} {kind}" for kind in ("hbo", "hbr") for c in record["ch_names"]],
                           record["sfreq"], ["hbo"] * n + ["hbr"] * n)
    haemo = mne.io.RawArray(record["data"].transpose(1, 0, 2).reshape(2 * n, -1), info, verbose=False)
    labelled = [(onset, record["event_labels"][name]) for onset, name in record["events"]
                if name in record["event_labels"]]
    if not labelled:
        return np.empty((0, n, 2, 0)), []
    onsets, labels = zip(*labelled)
    haemo.set_annotations(mne.Annotations(onsets, 0.0, labels))           # each event named by its label
    event_id = {label: code for code, label in enumerate(sorted(set(labels)), start=1)}
    events, _ = mne.events_from_annotations(haemo, event_id=event_id, verbose=False)
    windows = mne.Epochs(haemo, events, event_id, tmin, tmax, baseline=None, detrend=None, preload=True,
                         on_missing="ignore", verbose=False)
    name_of = {code: label for label, code in event_id.items()}
    X = windows.get_data().reshape(len(windows), 2, n, -1).transpose(0, 2, 1, 3)
    return X, [name_of[code] for code in windows.events[:, 2]]
