# Getting the data

The data are not redistributed. `datasets.csv` gives the download link (`data_source`) and the folder name
(`folder`) of every dataset.

1. Choose a data folder and set `FNIRS_DATA_ROOT` to it. Without it, the loaders look in `./data`.
   ```bash
   export FNIRS_DATA_ROOT=/path/to/fnirs_data
   ```
2. Download each dataset you need from its `data_source` into `$FNIRS_DATA_ROOT/<folder>`. Each loader's
   docstring says which files it reads inside that folder.
3. A few datasets need a one-off conversion or fix before they can be loaded. The original files are never modified.

| Dataset | Download to | Then run | Why |
|---|---|---|---|
| Resting state in AD and MCI | `cambridge_hddot_download/` | `python -m scripts.fix_cambridge_hddot resting` | SNIRF fields missing, so MNE cannot read the files |
| Cambridge visual task | `cambridge_visual_download/` | `python -m scripts.fix_cambridge_hddot visual` | same |
| Driving | `driving/` | `python -m scripts.recover_driving` | 25 truncated files |
| Hair and skin characteristics | `inclusion/` | `python -m scripts.fix_inclusion_sub400` | two unreadable files |
| Parent–child hyperscanning | `parents_download/` | `python -m scripts.convert_parents` | NIRStar files → SNIRF |
| Stroke BCI | `stroke_bci_download/` | `python -m scripts.fix_stroke_bci` | empty landmark entries |
| Motor recovery after stroke | `stroke_recovery_download/` | `python -m scripts.fix_stroke_recovery` | optical-density label, transposed data, three detector positions |
| Intracerebral hemorrhage | `hefmi_ich_download/source_data/` | `python -m scripts.convert_hefmi_ich` | Homer `.nirs` → SNIRF, with stimulus markers |

Loading the largest HD-DOT dataset (`vhddot_semantic_movies`) takes several hours.
