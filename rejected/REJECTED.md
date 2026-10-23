# Rejected datasets

These 24 datasets were reviewed but are **not** part of the pretraining corpus or the downstream tasks. Most lack
3D optode positions, so channel positions cannot be placed in MNI space. Their loaders are kept here for anyone who wants
the recordings anyway. They run the same processing as the main loaders (`loaders/utils.py`), but keep the positions
**exactly as provided**: native frame, no MNI conversion, `channel_xyz = None` when there are none.

Each loader reads `$FNIRS_DATA_ROOT/<folder>`, and its docstring gives the source and the reason for rejection.
Run one with `python -m rejected.load_<name>_data` from the repository root.

| Loader | Dataset | Source | Reason |
|---|---|---|---|
| `load_av_masking_data` | Audiovisual masking (Chen et al. 2025) | ScienceDB 10.57760/sciencedb.21750 | No positions; device HbO/HbR only |
| `load_buccino_data` | Hybrid EEG-fNIRS motor execution BCI (Buccino et al. 2016) | figshare 1619640, 1619641 | No positions: channels named by offsets from 10-20 electrodes |
| `load_chameleon_data` | Chameleon paradigm, Study 3 (Zhao et al. 2026) | ScienceDB 10.57760/sciencedb.25468 | Positions published for 12 of 24 channels only |
| `load_deception_data` | Social distance and deception hyperscanning (Huang 2024) | ScienceDB 10.57760/sciencedb.14965 | No positions |
| `load_hygrip_data` | HYGRIP grip force (Ortega et al. 2020) | figshare 12383639 | 2D polar offsets from C3/C4 only |
| `load_immigrants_data` | High-altitude immigrants Stroop (Ji et al. 2024) | figshare c.6979581 | No positions; device Hb only |
| `load_luhmann_resting_data` | Resting state, dataset I (von Lühmann et al. 2020) | NITRC group 1071 | 2D probe design only |
| `load_mindfulness_data` | Mindfulness and VR shooting (Gao & Zhang 2022) | Mendeley 84gbpd8s5s | 3D positions in an undocumented frame |
| `load_motion_data` | Motion artifacts (Yücel et al. 2014) | OpenNeuro ds005929 | 2D probe drawing only |
| `load_motion2_data` | Motion artifacts, dataset II (Yücel laboratory) | NITRC group 1071 | 2D probe design only |
| `load_multidistance_data` | Multi-distance ball squeezing (Gao et al. 2023) | OpenNeuro ds007420 | 3D positions in an undocumented frame, no landmarks |
| `load_nemo_data` | NEMO emotion (Spapé et al. 2024) | OSF pd9rv | Flat grid applied by the authors, no measured positions |
| `load_parkinson_data` | Parkinson's tapping, walking and rest (Guevara et al. 2024) | Zenodo 7966830 | 2D drawing only |
| `load_placebo_data` | Morphine and placebo pain (Peng et al. 2018) | OpenNeuro ds005777 | 2D layout only |
| `load_remifentanil_data` | Nociception under remifentanil (Karunakaran et al. 2022) | Harvard Dataverse 10.7910/DVN/RDDOW1 | 2D layout only; most files restricted |
| `load_rivalry_deng_data` | Binocular color rivalry (Deng 2025) | figshare 10.6084/m9.figshare.29253710 | 2D grid only |
| `load_spatial_attention_data` | Spatial attention (Ning et al. 2024) | OpenNeuro ds004830 | 2D probe design only |
| `load_stroop_block_data` | Block-design Stroop (Zeng 2025) | ScienceDB 10.57760/sciencedb.28229 | 2D grid only |
| `load_stroop_decoding_data` | Multi-day Stroop decoding (Zeng et al. 2026) | figshare 31839673 | 2D grid only |
| `load_stroop_eeg_data` | EEG+fNIRS Stroop (Chen et al. 2023) | figshare c.6623431 | No positions: 2D figure only |
| `load_tapping_data` | Finger and foot tapping (Bak et al. 2019) | figshare 9783755 | No positions: fields are NaN |
| `load_thermal_data` | Electrical, thermal and finger tapping (Yücel et al. 2015) | OpenNeuro ds005776 | 2D probe design only |
| `load_tinnitus_sanjuan_data` | Tinnitus resting state (San Juan et al. 2017) | Harvard Dataverse ZNZZBV | 2D layout only |
| `load_transdiagnostic_data` | Transdiagnostic verbal fluency (Wang et al. 2026) | figshare 32022984 | No positions |

Three more datasets were reviewed and have no loader, because their file formats are undocumented or encrypted:
- NIRSIT normal and stroke (Mendeley 63rszpyfns)
- NIRSIT hand movements (Mendeley jhv73gj2d2)
- Binocular color fusion and rivalry (figshare 27330858)
