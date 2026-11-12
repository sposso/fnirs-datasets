"""Write the Hugging Face dataset card (README.md) and meta/datasets.parquet from datasets.csv.

    python scripts/make_hf_card.py --out hf [--references paper/references.bib]

Channel counts and recording counts come from ``<out>/meta/recordings/`` (datasets already exported).
``--references`` takes the paper's BibTeX file and stores, in ``<out>/.citations.bib``, the entries listed in
the ``bib_keys`` column of datasets.csv (each dataset's paper and data record, as cited in the paper); the card
embeds that file.
"""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

REPO = "sposso22/fnirs-datasets"
GITHUB = "https://github.com/sposso/fnirs-datasets"
PAPER = "https://arxiv.org/abs/2610.40277"
PAPER_BIB = """@article{possomurillo2026fnirscorpus,
  title={Towards fNIRS foundation models: A curated corpus of 67 public datasets with channel positions in MNI space},
  author={Posso-Murillo, Santiago},
  journal={arXiv preprint arXiv:2610.40277},
  year={2026}
}"""


def hub_license(original):
    """CC BY-SA sets must stay CC BY-SA; everything else is shared under CC BY-NC-SA."""
    return "CC BY-SA 4.0" if original == "CC BY-SA 4.0" else "CC BY-NC-SA 4.0"


def paper_entries(rows, references, path):
    """Copy each dataset's entries (``bib_keys`` in datasets.csv) from the paper's BibTeX file, unchanged."""
    text = Path(references).read_text()
    entries = {m.group(1): e.strip() for e in re.split(r"(?=@\w+\s*\{[^,\s]+,)", text)   # entries may share a line
               if (m := re.match(r"@\w+\s*\{([^,\s]+),", e))}
    keys = list(dict.fromkeys(k for r in rows for k in r["bib_keys"].split(";")))
    missing = [k for k in keys if k not in entries]
    if missing:
        raise SystemExit(f"not in {references}: {', '.join(missing)}")
    path.write_text("\n\n".join(re.sub(r"\}[^}]*$", "}", entries[k]) for k in keys) + "\n")
    print(f"{len(keys)} BibTeX entries for {len(rows)} datasets")


def datasets_table(rows, out):
    table = pd.DataFrame(rows)
    table["participants"] = table["participants"].astype(int)
    table["hours"] = table["hours"].astype(float)
    table["license_here"] = table["license"].map(hub_license)
    files = sorted((out / "meta" / "recordings").glob("*.parquet"))
    if files:
        recordings = pd.concat([pd.read_parquet(f) for f in files])
        stats = recordings.groupby("dataset").agg(recordings=("recording_id", "size"),
                                                  channels_min=("n_channels", "min"),
                                                  channels_max=("n_channels", "max"))
        table = table.join(stats, on="name")
    (out / "meta").mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "meta" / "datasets.parquet", index=False)
    return table


def card(table, bib):
    def channels(r):
        if pd.isna(r.get("channels_min")):
            return ""
        lo, hi = int(r["channels_min"]), int(r["channels_max"])
        return str(lo) if lo == hi else f"{lo}–{hi}"

    def rows(role):
        lines = ["| Folder | Dataset | Participants | Hours | Channels | License | Source |",
                 "|---|---|---:|---:|---:|---|---|"]
        for _, r in table[table["role"] == role].iterrows():
            source = " · ".join(f"[data]({u.strip()})" for u in r["data_source"].split(";"))
            if r["paper"]:
                source += f" · [paper]({r['paper']})"
            lines.append(f"| `{r['name']}` | {r['dataset']} | {r['participants']} | {r['hours']:.1f} | "
                         f"{channels(r)} | {r['license']} | {source} |")
        return "\n".join(lines)

    n_rec = int(table["recordings"].sum()) if "recordings" in table else 0
    size = "1K<n<10K" if n_rec < 10_000 else "10K<n<100K"
    by_sa = ", ".join(f"`{n}`" for n in table.loc[table["license_here"] == "CC BY-SA 4.0", "name"])
    pre, down = table[table["role"] == "pretraining"], table[table["role"] == "downstream"]

    return f"""---
license:
- cc-by-nc-sa-4.0
- cc-by-sa-4.0
pretty_name: fNIRS datasets
size_categories:
- {size}
tags:
- fnirs
- neuroimaging
- time-series
configs:
- config_name: all
  default: true
  data_files: data/*/*.parquet
- config_name: datasets
  data_files: meta/datasets.parquet
- config_name: recordings
  data_files: meta/recordings/*.parquet
---

# fNIRS datasets

{len(table)} public functional near-infrared spectroscopy (fNIRS) datasets in a common format, with channel
positions in MNI space: {table['hours'].sum():,.0f} hours from {table['participants'].sum():,} participants
({len(pre)} pretraining datasets, {len(down)} downstream datasets).

- Paper: {PAPER}
- Code: {GITHUB}

## Load the dataset

```python
from datasets import load_dataset

ds = load_dataset("{REPO}", split="train")                                            # all recordings
ds = load_dataset("{REPO}", data_files="data/fresh_motor/*.parquet", split="train")  # one dataset
```

Subsets: `all` (default; one row per recording), `datasets` (one row per dataset), `recordings` (one row per
recording, without signals).

## Data fields

| Field | Description |
|---|---|
| `recording_id`, `dataset`, `subject_id` | identifiers |
| `hbo`, `hbr` | HbO and HbR concentration changes (mol/L), `(channels, samples)` at `sfreq` (7.5 Hz) |
| `ch_names` | channel names, e.g. `S1_D1` |
| `channel_xyz` | channel positions (source–detector midpoints), MNI, meters |
| `source_xyz`, `detector_xyz` | optode positions, MNI, meters; NaN when only channel positions are available |
| `sd_distances` | source–detector distances, meters |
| `events` | `onset` (s) and `name` of each marker |
| `raw_signal` | signal as stored in the source file, `(rows, samples)` at `raw_sfreq` |
| `raw_names` | `raw_signal` row names: channel and wavelength (`S1_D1 760`) or chromophore (`S1_D1 hbo`) |
| `raw_kind` | `intensity`, `od` (optical density) or `hb` (HbO/HbR from the device software) |

## Processing

`hbo` and `hbr`: light intensity → optical density → modified Beer–Lambert law (partial pathlength factor 6) →
0.01 Hz high-pass → resampling to 7.5 Hz. Datasets that share only device-computed HbO/HbR (`raw_kind` = `hb`)
get the last two steps. No motion correction and no quality-based channel rejection.

Removed: channels flagged as bad by the original authors, participants or runs excluded in the data record,
channels with missing-data gaps longer than 1 s, and source–detector pairs longer than 35 mm in high-density
arrays. The conversion of each dataset's positions to MNI space is documented in the code repository.

## Datasets

### Pretraining

{rows("pretraining")}

### Downstream

{rows("downstream")}

## References to the original datasets

<details>
<summary>BibTeX</summary>

```bibtex
{bib.strip()}
```

</details>

## Licensing information

CC BY-NC-SA 4.0, except {by_sa}, which keep their original CC BY-SA 4.0 license. The original license of each
dataset is listed in the tables above. `lexical_retest` excludes sub-012 and sub-022 (no consent to secondary
use); its other conditions are in the original data record.

## Citation information

```bibtex
{PAPER_BIB}
```

Please also cite the original datasets you use.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="hf")
    parser.add_argument("--references", help="the paper's BibTeX file")
    args = parser.parse_args()
    out = Path(args.out)
    rows = list(csv.DictReader(open(Path(__file__).parents[1] / "datasets.csv")))
    bib_path = out / ".citations.bib"
    if args.references:
        paper_entries(rows, args.references, bib_path)
    (out / "README.md").write_text(card(datasets_table(rows, out), bib_path.read_text()))
    print(f"wrote {out}/README.md and {out}/meta/datasets.parquet")
