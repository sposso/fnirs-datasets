"""Export one dataset's records to Parquet for the Hugging Face Hub, and check the files against the records.

One row per recording, with every field of the record (see the README). Arrays are stored as nested lists of
float32; ``hbo`` and ``hbr`` are the two halves of ``data``. Writes

    <out>/data/<name>/NNNN.parquet         recordings, shards of about 500 MB, one row group per recording
    <out>/meta/recordings/<name>.parquet   one row per recording without the signals (the "recordings" subset)

and then reads the shards back and compares every array with the loader's record (float32 precision).

    python scripts/export_hf.py fresh_motor --out hf
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from loaders import load

SHARD_BYTES = 500e6
VIEWER_LIMIT = 286 * 2**20   # largest row group the Hugging Face dataset viewer reads

SCHEMA = pa.schema([
    ("recording_id", pa.string()),
    ("dataset", pa.string()),
    ("subject_id", pa.string()),
    ("sfreq", pa.float64()),
    ("ch_names", pa.list_(pa.string())),
    ("hbo", pa.list_(pa.list_(pa.float32()))),           # (C, T), M
    ("hbr", pa.list_(pa.list_(pa.float32()))),           # (C, T), M
    ("channel_xyz", pa.list_(pa.list_(pa.float32()))),   # (C, 3), MNI, metres
    ("source_xyz", pa.list_(pa.list_(pa.float32()))),    # (C, 3), MNI, metres; NaN if not given
    ("detector_xyz", pa.list_(pa.list_(pa.float32()))),  # (C, 3)
    ("sd_distances", pa.list_(pa.float32())),            # (C,), metres
    ("events", pa.list_(pa.struct([("onset", pa.float64()), ("name", pa.string())]))),
    ("raw_signal", pa.list_(pa.list_(pa.float32()))),    # (M, T_raw), signal as read from the file
    ("raw_names", pa.list_(pa.string())),
    ("raw_sfreq", pa.float64()),
    ("raw_kind", pa.string()),                           # "intensity", "od" or "hb"
])
ARRAYS = ["hbo", "hbr", "channel_xyz", "source_xyz", "detector_xyz", "sd_distances", "raw_signal"]


def to_row(name, recording_id, r):
    f32 = lambda a: np.asarray(a, dtype=np.float32).tolist()
    return {
        "recording_id": recording_id,
        "dataset": name,
        "subject_id": str(r["subject_id"]),
        "sfreq": float(r["sfreq"]),
        "ch_names": list(r["ch_names"]),
        "hbo": f32(r["data"][:, 0]),
        "hbr": f32(r["data"][:, 1]),
        "channel_xyz": f32(r["channel_xyz"]),
        "source_xyz": f32(r["source_xyz"]),
        "detector_xyz": f32(r["detector_xyz"]),
        "sd_distances": f32(r["sd_distances"]),
        "events": [{"onset": float(t), "name": str(e)} for t, e in r["events"]],
        "raw_signal": f32(r["raw_signal"]),
        "raw_names": list(r["raw_names"]),
        "raw_sfreq": float(r["raw_sfreq"]),
        "raw_kind": r["raw_kind"],
    }


def meta_row(row, file):
    return {
        "recording_id": row["recording_id"],
        "dataset": row["dataset"],
        "subject_id": row["subject_id"],
        "n_channels": len(row["ch_names"]),
        "duration_s": len(row["hbo"][0]) / row["sfreq"],
        "sfreq": row["sfreq"],
        "raw_kind": row["raw_kind"],
        "raw_sfreq": row["raw_sfreq"],
        "n_events": len(row["events"]),
        "event_names": sorted({e["name"] for e in row["events"]}),
        "file": file,
    }


def check(records, folder):
    """Read the shards back and compare them with the records; return the largest row group in bytes."""
    n, largest = 0, 0
    for path in sorted(folder.glob("*.parquet")):
        file = pq.ParquetFile(path)
        largest = max(largest, *(file.metadata.row_group(i).total_byte_size for i in range(file.num_row_groups)))
        for row in file.read().to_pylist():
            r = records[row["recording_id"]]
            expected = {"hbo": r["data"][:, 0], "hbr": r["data"][:, 1], **{k: r[k] for k in ARRAYS[2:]}}
            for key in ARRAYS:
                # float32 is exact to ~6e-8 relative, except below its smallest normal value (1.2e-38)
                assert np.allclose(np.array(row[key], np.float64), expected[key], rtol=1e-6,
                                   atol=np.finfo(np.float32).tiny, equal_nan=True), \
                    f"{row['recording_id']}: {key} differs"
            assert row["ch_names"] == list(r["ch_names"]) and row["raw_names"] == list(r["raw_names"])
            assert [(e["onset"], e["name"]) for e in row["events"]] == [(float(t), str(e)) for t, e in r["events"]]
            n += 1
    assert n == len(records), f"{n} rows for {len(records)} records"
    return largest


def export(name, out):
    out = Path(out)
    folder = out / "data" / name
    folder.mkdir(parents=True, exist_ok=True)
    (out / "meta" / "recordings").mkdir(parents=True, exist_ok=True)
    records = load(name)
    rows, meta, size, shard = [], [], 0, 0

    def flush():
        nonlocal rows, size, shard
        file = f"data/{name}/{shard:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out / file, row_group_size=1, compression="zstd")
        meta.extend(meta_row(row, file) for row in rows)
        rows, size, shard = [], 0, shard + 1

    for recording_id, record in records.items():
        rows.append(to_row(name, recording_id, record))
        size += record["data"].nbytes / 2 + record["raw_signal"].nbytes / 2   # float64 -> float32
        if size > SHARD_BYTES:
            flush()
    if rows:
        flush()
    pq.write_table(pa.Table.from_pylist(meta), out / "meta" / "recordings" / f"{name}.parquet")

    largest = check(records, folder)
    print(f"{name}: {len(records)} recordings in {shard} shard(s), all match the loader; "
          f"largest row group {largest / 2**20:.0f} MB" + ("  ** above the viewer limit **" if largest > VIEWER_LIMIT else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--out", default="hf")
    args = parser.parse_args()
    export(args.name, args.out)
