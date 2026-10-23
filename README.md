# fnirs-datasets

Loaders for 67 public fNIRS datasets in one common format with channel positions in MNI space: 1,954.84 hours of
recordings from 2,798 participants. Each recording becomes HbO and HbR concentration changes at 7.5 Hz, with the MNI
position of every channel. Of the 67, 61 datasets form a pretraining corpus and 6 are held out as downstream tasks
with labeled epochs. The repository gives a download link for every dataset.

The goal is to make foundation models for fNIRS easier to build: pretraining needs many datasets pooled together,
and pooling fNIRS datasets needs a common format and a common spatial reference for channels. This repository
provides both, together with a fixed benchmark to compare models on.

- **Pretraining corpus:** 61 datasets, 2,577 participants, 1,835.88 hours.
- **Downstream tasks:** 6 held-out datasets, 221 participants, 118.96 hours, with labeled epochs.

> Paper: *coming soon*. Please cite it, and every dataset you use (see [Citation](#citation)).

## Corpus overview

| Category | Datasets | Participants | Hours |
|---|---:|---:|---:|
| Social interaction | 7 | 514 | 486.55 |
| Motor | 15 | 381 | 416.80 |
| Cognitive | 14 | 491 | 363.81 |
| Speech | 11 | 644 | 310.62 |
| Resting state | 6 | 353 | 132.61 |
| Visual | 5 | 92 | 67.03 |
| Emotion | 3 | 102 | 58.46 |
| **Total** | **61** | **2,577** | **1,835.88** |

`datasets.csv` lists every dataset with its category, population, participants, hours, download link and paper.

## Installation

```bash
git clone https://github.com/sposso/fnirs-datasets.git && cd fnirs-datasets
pip install -r requirements.txt
```

## Getting the data

The data are not redistributed. `datasets.csv` gives the download link of every dataset. Download each dataset
into `$FNIRS_DATA_ROOT/<folder>`; see [docs/getting_data.md](docs/getting_data.md) for the few datasets that need
a conversion step first.

## Usage

```python
from loaders import load

records = load("fresh_motor")            # {recording_id: record}
recording_id, record = next(iter(records.items()))
record["data"].shape                     # (channels, 2, samples): [HbO, HbR] in M, 7.5 Hz
```

The dataset names are the keys of `loaders.PRETRAINING` and `loaders.DOWNSTREAM`. Each record holds:

| Key | Content |
|---|---|
| `data` | `(C, 2, T)` array: HbO and HbR concentration changes (M) of `C` channels at 7.5 Hz |
| `channel_xyz` | `(C, 3)` channel positions (source-detector midpoints) in MNI space, meters |
| `sd_distances` | `(C,)` source-detector distances, meters |
| `ch_names` | channel names, e.g. `"S1_D1"` |
| `events` | `[(onset s, name), ...]` |
| `subject_id` | participant identifier |

Downstream records add two keys: `epochs`, an `(n, C, 2, T)` array with one epoch per trial of the task, and
`labels`, the class of each epoch (for example `animal` or `tool`). See `examples/downstream_epochs.py`.

## Processing

The same minimal pipeline for every dataset: optical density → HbO/HbR (modified Beer–Lambert law), high-pass
filter at 0.01 Hz, resampling to 7.5 Hz. No channel is rejected for signal quality and no motion correction is
applied. Details, including how each dataset's positions were brought to MNI space, are in
[docs/processing.md](docs/processing.md) and in each loader's docstring.

## Downstream tasks

| Task | Classes | Epoch | Name |
|---|---|---|---|
| ICH | healthy vs. ICH | 10-s motor imagery | `hefmi_ich` |
| CI status | normal hearing vs. cochlear implant | 13-s sentence | `speech_comprehension` |
| Tinnitus | control vs. tinnitus | 12-s noise block | `tinnitus` |
| Movement | movement vs. no movement | 12 s from each LED flash | `stroke_recovery` |
| Action mode | observation vs. execution | 4-s action trial | `mirror` |
| Semantic category | animal vs. tool | 14.3-s trial | `tools_animals` |

## Rejected datasets

24 more datasets were reviewed and not used, mainly for lacking 3D optode positions. Their loaders and the reasons
are in [rejected/REJECTED.md](rejected/REJECTED.md).

## Datasets

Every dataset with its paper and its data record. The participants and hours of each one are in `datasets.csv`.

### Pretraining

1. **Attention tasks after brain injury** — Cortical activation and functional connectivity during different attention tasks using functional near-infrared spectroscopy in middle-aged and elderly people (Ding et al., 2025). [Paper](https://doi.org/10.1186/s40001-025-02597-1) · [Data](https://doi.org/10.17632/7tcrj59w2p.1)
2. **Post-error multitasking** — The Neurophysiological Mechanisms of Post-Error Responses in Multitasking Scenarios (Zhu et al., 2025). [Data](https://doi.org/10.57760/sciencedb.28380)
3. **Word–color Stroop** — High-density multidistance fNIRS enhances detection of brain activity during a word-color Stroop task (Anderson et al., 2025). [Paper](https://doi.org/10.1117/1.NPh.12.3.035010) · [Data](https://doi.org/10.18112/openneuro.ds006460.v1.0.0) · [Data](https://doi.org/10.18112/openneuro.ds006459.v1.0.1)
4. **Visuomotor target identification** — Simultaneous Dataset of Brain, Eye and Hand during Visuomotor Tasks (Zhang et al., 2025). [Paper](https://doi.org/10.1038/s41597-024-04227-7) · [Data](https://doi.org/10.17605/OSF.IO/CFDSZ)
5. **AI persuasion** — Wired to believe: source-authority cues shape perceived AI capability and neural persuasion (Ding et al., 2026). [Paper](https://doi.org/10.1080/00140139.2026.2734930) · [Data](https://doi.org/10.57760/sciencedb.42071)
6. **Real use of a cosmetic product** — Willingness-to-pay-associated right prefrontal activation during a single, real use of cosmetics as revealed by functional near-infrared spectroscopy (Kawabata Duncan et al., 2019). [Paper](https://doi.org/10.3389/fnhum.2019.00016) · [Data](https://doi.org/10.6084/m9.figshare.7409834)
7. **Online rumors** — Authority reliance vs. deliberative assessment in processing online rumors: evidence from fNIRS (Cheng et al., 2026). [Paper](https://doi.org/10.3389/fnhum.2026.1750499) · [Data](https://doi.org/10.57760/sciencedb.35305)
8. **Cognitive and motor tasks** — Multimodal dataset from the CMx7-MM Experiment (Ajra et al., 2026). [Data](https://doi.org/10.18112/openneuro.ds007554.v1.0.0)
9. **Automated driving** — An fNIRS dataset for driving risk cognition of passengers in highly automated driving scenarios (Zhang et al., 2024). [Paper](https://doi.org/10.1038/s41597-024-03353-6) · [Data](https://doi.org/10.18112/openneuro.ds004973.v1.0.1)
10. **Simulated flights (novices)** — Investigating mental workload-induced changes in cortical oxygenation and frontal theta activity during simulated flights. Scientific Reports 12, 6449. doi:10.1038/s41598-022-10044-y. Data: PsychArchives doi:10.23668/psycharchives.5291 (fnirs_data_stims.mat) (Hamann & Carstengerdes, 2022). [Paper](https://doi.org/10.1038/s41598-022-10044-y) · [Data](https://doi.org/10.23668/psycharchives.5291)
11. **Simulated flights (pilots)** — Examining physiological parameters of mental workload in the cockpit: a multimethod approach (Hamann et al., 2026). [Paper](https://doi.org/10.3389/fphys.2026.1893570) · [Data](https://doi.org/10.17605/OSF.IO/YXRKE)
12. **n-back** — Increased cognitive workload evokes greater neurovascular coupling responses in healthy young adults (Csipo et al., 2021). [Paper](https://doi.org/10.1371/journal.pone.0250043) · [Data](https://doi.org/10.13026/zfb2-1g43)
13. **Working memory, discrimination, word generation** — Simultaneous acquisition of EEG and NIRS during cognitive tasks for an open access dataset (Shin et al., 2018). [Paper](https://doi.org/10.1038/sdata.2018.3) · [Data](https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/)
14. **Symmetry in visual working memory** — Neurocognitive dynamics and behavioral differences of symmetry and asymmetry processing in working memory: insights from fNIRS. Scientific Reports 15, 4740. doi:10.1038/s41598-024-84988-8. Data: (Sztuka & Kühn, 2025). [Paper](https://doi.org/10.1038/s41598-024-84988-8) · [Data](https://doi.org/10.17605/OSF.IO/Z7KW3)
15. **Empathy and Pipa music** — The effect of empathy on emotional response to Pipa music: evidence from fNIRS [in Chinese] (Ge et al., 2024). [Paper](https://doi.org/10.12074/202411.00062) · [Data](https://doi.org/10.57760/sciencedb.14924)
16. **Emotional film clips** — REFED: A Subject Real-time Dynamic Labeled EEG-fNIRS Synchronized Recorded Emotion Dataset (Ning et al., 2025). [Paper](https://neurips.cc/virtual/2025/poster/121785) · [Data](https://huggingface.co/datasets/REFED2025/REFED-dataset)
17. **Detached and positive reappraisal** — Differential neural correlates of two cognitive reappraisal strategies: evidence from functional near-infrared spectroscopy (Wang et al., 2025). [Paper](https://doi.org/10.2139/ssrn.5163908) · [Data](https://doi.org/10.57760/sciencedb.13806)
18. **Exercise-induced hypoalgesia** — Profound neuronal differences during Exercise-Induced Hypoalgesia between athletes and non-athletes revealed by functional near-infrared spectroscopy (Geisler et al., 2025). [Data](https://doi.org/10.18112/openneuro.ds006902.v1.1.1)
19. **Freezing of gait in Parkinson's disease** — Freezing of gait in Parkinson's disease is related to imbalanced stopping-related cortical activity (Cockx et al., 2024). [Paper](https://doi.org/10.1093/braincomms/fcae259) · [Data](https://doi.org/10.34973/k7ce-6n58)
20. **Treadmill walking** — Enhancing classification accuracy of fNIRS-BCI for gait rehabilitation (Minhas et al., 2024). [Paper](https://doi.org/10.1109/ACCESS.2024.3443066) · [Data](https://doi.org/10.6084/m9.figshare.31871557)
21. **Ball squeezing (whole-head HD)** — Surface-based image reconstruction optimization for high-density functional near-infrared spectroscopy (Carlton et al., 2026). [Paper](https://doi.org/10.1117/1.NPh.13.2.025001) · [Data](https://openneuro.org/datasets/ds006673/versions/1.0.4)
22. **Finger and leg movements** — fNIRS is sensitive to leg activity in the primary motor cortex after systemic artifact correction (Cockx et al., 2023). [Paper](https://doi.org/10.1016/j.neuroimage.2023.119880) · [Data](https://doi.org/10.34973/vesb-mh30)
23. **Individual finger tapping** — Open access individual finger movement dataset with fNIRS (Khan et al., 2026). [Paper](https://doi.org/10.3389/fnhum.2026.1747655) · [Data](https://figshare.com/s/6298861b1dc6be936d73?file=59576999)
24. **Motor task** — FRESH Motor Dataset (Mesquita et al., 2025). [Paper](https://doi.org/10.1038/s42003-025-08412-1) · [Data](https://doi.org/10.18112/openneuro.ds005963.v1.0.0)
25. **Unilateral finger tapping** — fNIRS Finger Tapping Data in BIDS Format (v0.0.1) (Luke & McAlpine, 2021). [Data](https://doi.org/10.5281/zenodo.5529797)
26. **Balance tasks** — Observation and motor imagery balance tasks evaluation: An fNIRS feasibility study (Almulla et al., 2022). [Paper](https://doi.org/10.1371/journal.pone.0265898) · [Data](https://doi.org/10.6084/m9.figshare.19131281.v1)
27. **Motor imagery of upper-limb joints** — A multi-modal dataset of electroencephalography and functional near-infrared spectroscopy recordings for motor imagery of multi-types of joints from unilateral upper limb (Yi et al., 2025). [Paper](https://doi.org/10.1038/s41597-025-05286-0) · [Data](https://doi.org/10.6084/m9.figshare.24123303)
28. **Directional motor imagery of the right arm** — A Multimodal fNIRS-EEG Dataset for Unilateral Limb Motor Imagery (Feng et al., 2026). [Paper](https://doi.org/10.1038/s41597-026-07807-x) · [Data](https://doi.org/10.57760/sciencedb.34326)
29. **Motor imagery and mental arithmetic** — Open access dataset for EEG+NIRS single-trial classification (Shin et al., 2017). [Paper](https://doi.org/10.1109/TNSRE.2016.2628057) · [Data](https://doc.ml.tu-berlin.de/hBCI/)
30. **BCI training after stroke** — A multiple session dataset of NIRS recordings from stroke patients controlling brain-computer interface (Isaev et al., 2024). [Paper](https://doi.org/10.1038/s41597-024-04012-6) · [Data](https://doi.org/10.25790/bml0cm.167)
31. **Laparoscopic training across a 24-h shift** — Resident physician mental workload assessed with fNIRS (Torres-Cuevas et al., 2025). [Data](https://doi.org/10.5281/zenodo.15186569)
32. **Laparoscopic surgical training** — Assessment of Surgical Tasks Using Neuroimaging Dataset (ASTaUND) (Kamat et al., 2023). [Paper](https://doi.org/10.1038/s41597-023-02603-3) · [Data](https://doi.org/10.6084/m9.figshare.c.6429713)
33. **Resting state in AD and MCI** — Brain network analysis in Alzheimer's disease and mild cognitive impairment using high-density diffuse optical tomography (Butters et al., 2026). [Paper](https://doi.org/10.1162/IMAG.a.1208) · [Data](https://doi.org/10.18112/openneuro.ds007719.v1.0.0)
34. **Hair and skin characteristics** — Quantifying the impact of hair and skin characteristics on fNIRS signal quality for enhanced inclusivity. Nature Human Behaviour 9, 2651-2668. doi:10.1038/s41562-025-02274-7 (Yücel et al., 2025). [Paper](https://doi.org/10.1038/s41562-025-02274-7) · [Data](https://doi.org/10.18112/openneuro.ds006377.v1.0.2)
35. **Yoga intervention** — Multimodal Dataset of a 30-Day Yoga Intervention: fNIRS, Biochemical, Postural, and Subjective Outcomes (Goodrick et al., 2026). [Data](https://doi.org/10.5281/zenodo.20560768)
36. **General anesthesia** — Multimodal EEG-fNIRS data from patients undergoing general anesthesia (Saini Ferrón et al., 2026). [Data](https://doi.org/10.18112/openneuro.ds004541.v1.0.0)
37. **Infant resting state** — Open access dataset of task-free hemodynamic activity in 4-month-old infants during sleep using fNIRS (Blanco et al., 2022). [Paper](https://doi.org/10.1038/s41597-022-01210-y) · [Data](https://doi.org/10.17605/OSF.IO/7FZKM)
38. **Bedside resting state after stroke** — Disrupted functional connectivity during stroke recovery revealed via bedside optical neuroimaging (Bergonzi et al., 2026). [Paper](https://doi.org/10.1117/1.NPh.13.3.035002) · [Data](https://doi.org/10.18112/openneuro.ds008798.v2.1.0)
39. **Natural conversations about appearance** — Dataset of natural conversations about appearance using fNIRS (Hu et al., 2025). [Paper](https://doi.org/10.1038/s41597-025-05574-9) · [Data](https://doi.org/10.17605/OSF.IO/HMVSU)
40. **Cooperative visuomotor control in dyads** — A dyadic fNIRS hyperscanning dataset for cooperative visuomotor control with haptic interaction (Zheng et al., 2026). [Data](https://doi.org/10.57760/sciencedb.46254)
41. **Online face gaze** — Investigating face processing in online interactions via UK-US hyperscanning using fNIRS (Hakim et al., 2026). [Paper](https://doi.org/10.1162/IMAG.a.1101) · [Data](https://doi.org/10.5061/dryad.2ngf1vj10)
42. **Intergenerational collaborative drawing** — A longitudinal fNIRS hyperscanning dataset of same generation and intergenerational relationship development and collaboration (Moffat & Cross, 2026). [Paper](https://doi.org/10.64898/2026.06.26.734715) · [Data](https://doi.org/10.18112/openneuro.ds008192.v1.0.5)
43. **Social touch** — An fNIRS hyperscanning dataset on the modulation of synchrony by social relationship during social touch (Chen et al., 2025). [Paper](https://doi.org/10.1038/s41597-025-06233-9) · [Data](https://doi.org/10.17605/OSF.IO/SM7YT)
44. **Parent–child interaction** — Dataset of parent-child hyperscanning functional near-infrared spectroscopy recordings (Bizzego et al., 2022). [Paper](https://doi.org/10.1038/s41597-022-01751-2) · [Data](https://doi.org/10.21979/N9/35DNCW)
45. **Co-parenting couples** — Physical presence of spouse enhances brain-to-brain synchrony in co-parenting couples. Scientific Reports 10, 7569. doi:10.1038/s41598-020-63596-2. Data: DR-NTU doi:10.21979/N9/KF1JOG (CC BY-NC 4.0) (Azhari et al., 2020). [Paper](https://doi.org/10.1038/s41598-020-63596-2) · [Data](https://doi.org/10.21979/N9/KF1JOG)
46. **Passive listening** — FRESH Audio Dataset (Luke et al., 2025). [Paper](https://doi.org/10.1038/s42003-025-08412-1) · [Data](https://doi.org/10.18112/openneuro.ds005964.v1.0.0)
47. **Cocktail-party spatial attention** — Decoding spatial attention in the cocktail party problem using wearable whole-head high-density fNIRS (Duwadi et al., 2026). [Paper](https://doi.org/10.64898/2026.05.06.722322) · [Data](https://doi.org/10.18112/openneuro.ds007738.v1.0.0)
48. **Interaural coherence of sounds** — Human cortical processing of interaural coherence (Luke et al., 2022). [Paper](https://doi.org/10.1016/j.isci.2022.104181) · [Data](https://osf.io/tyzag/)
49. **Voice pitch changes** — Cortical activity evoked by voice pitch changes: A combined fNIRS and EEG study (Steinmetzger et al., 2022). [Paper](https://doi.org/10.1016/j.heares.2022.108483) · [Data](https://osf.io/ps47b)
50. **Naturalistic story listening** — Neural processing of children's theory of mind in a naturalistic story-listening paradigm (Yu et al., 2025). [Paper](https://doi.org/10.1093/scan/nsaf022) · [Data](https://doi.org/10.7302/zw4w-hp67)
51. **Auditory-only and visual-only speech** — The use of broad vs restricted regions of interest in functional near-infrared spectroscopy for measuring cortical activation to auditory-only and visual-only speech (Shader et al., 2021). [Paper](https://doi.org/10.1016/j.heares.2021.108256) · [Data](https://osf.io/xwerv)
52. **Test–retest lexical decision** — Investigating fNIRS test-retest reliability during lexical decision (Elliott et al., 2026). [Paper](https://doi.org/10.64898/2026.07.14.737938) · [Data](https://doi.org/10.5683/SP4/MXMU9I)
53. **Morphological and phonological processing** — Morphological and phonological processing in English monolingual, Chinese-English bilingual, and Spanish-English bilingual children: An fNIRS neuroimaging dataset (Sun et al., 2022). [Paper](https://doi.org/10.1016/j.dib.2022.108048) · [Data](https://doi.org/10.7302/gg8n-mq88)
54. **Sentence-level prosodic processing** — The development of left hemisphere lateralization for sentence-level prosodic processing (Pasquinelli et al., 2023). [Paper](https://doi.org/10.1044/2022_JSLHR-22-00103) · [Data](https://doi.org/10.7302/xj95-gv44)
55. **Relative clause processing in Turkish** — Functional near-infrared spectroscopy shows that object relative clauses are more difficult to process than subject relative clauses in Turkish (Mutlu et al., 2023). [Paper](https://doi.org/10.1111/ejn.15930) · [Data](https://doi.org/10.6084/m9.figshare.21923175.v1)
56. **Covert and overt language production** — Mapping cortical activations underlying covert and overt language production using high-density diffuse optical tomography (Schroeder et al., 2023). [Paper](https://doi.org/10.1016/j.neuroimage.2023.120190) · [Data](https://www.nitrc.org/frs/?group_id=1563)
57. **Animated movies and localizers (HD-DOT)** — Mapping brain function in adults and young children during naturalistic viewing with high-density diffuse optical tomography (Tripathy et al., 2024). [Paper](https://doi.org/10.1002/hbm.26684) · [Data](https://doi.org/10.18112/openneuro.ds007473.v1.0.0)
58. **Whole-head VHD-DOT localizers and movies** — Functional brain mapping using whole-head very high-density diffuse optical tomography (Fogarty et al., 2025). [Paper](https://doi.org/10.1162/IMAG.a.54) · [Data](https://doi.org/10.18112/openneuro.ds007463.v1.1.1)
59. **Visual semantics of movies (VHD-DOT)** — Visual semantic encoding and identification of naturalistic movies via high-density diffuse optical tomography (Fehner et al., 2025). [Paper](https://doi.org/10.64898/2025.12.03.692158) · [Data](https://doi.org/10.18112/openneuro.ds007816.v1.1.1)
60. **Visual task in AD and MCI** — Brain network analysis in Alzheimer's disease and mild cognitive impairment using high-density diffuse optical tomography (Butters et al., 2026). [Paper](https://doi.org/10.1162/IMAG.a.1208) · [Data](https://doi.org/10.18112/openneuro.ds007714.v1.0.0)
61. **Calibrated fMRI** — A comparison of continuous-wave fNIRS with quantitative fMRI-derived indices of brain function in the visual cortex (Rocco et al., 2026). [Paper](https://doi.org/10.64898/2026.08.10.743546) · [Data](https://gin.g-node.org/giuliarocco/DOMRIBrain)

### Downstream tasks

1. **Action observation and execution** — Brain correlates of motor complexity during observed and executed actions. Scientific Reports 10, 10965. doi:10.1038/s41598-020-67327-5 (Li et al., 2020). [Paper](https://doi.org/10.1038/s41598-020-67327-5) · [Data](https://doi.org/10.18112/openneuro.ds005935.v1.0.0)
2. **Motor recovery after stroke** — Multisession fNIRS-EEG data of post-stroke motor recovery. Recordings during intact and paretic hand movements (Medvedeva et al., 2026). [Paper](https://doi.org/10.1038/s41597-026-06803-5) · [Data](https://doi.org/10.6084/m9.figshare.28904942)
3. **Intracerebral hemorrhage** — HEFMI-ICH: a hybrid EEG-fNIRS motor imagery dataset for brain-computer interface in intracerebral hemorrhage (Shi et al., 2025). [Paper](https://doi.org/10.1038/s41597-025-06100-7) · [Data](https://doi.org/10.6084/m9.figshare.28955456.v4)
4. **Auditory stimulation in tinnitus** — Subjective tinnitus severity correlates with objective changes in human auditory cortex as measured by functional near-infrared spectroscopy (Satish et al., 2026). [Data](https://doi.org/10.18112/openneuro.ds007990.v1.0.0)
5. **Speech comprehension with cochlear implants** — An fNIRS dataset for Multimodal Speech Comprehension in Normal Hearing Individuals and Cochlear Implant Users (Bálint et al., 2025). [Paper](https://doi.org/10.1038/s41597-025-05654-w) · [Data](https://doi.org/10.5061/dryad.crjdfn3g9)
6. **Semantic imagery of animals and tools** — Simultaneous EEG and fNIRS recordings for semantic decoding of imagined animals and tools (Rybář et al., 2025). [Paper](https://doi.org/10.1038/s41597-025-04967-0) · [Data](https://doi.org/10.18112/openneuro.ds004514.v1.1.2)

## Citation

Please cite our paper (BibTeX coming soon) and each dataset you use; the data record and paper of every dataset are
in `datasets.csv`.

## License

The code is under the MIT License. The datasets keep the licenses of their original repositories.
