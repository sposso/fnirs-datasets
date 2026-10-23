"""MNI optode positions for the surgical dataset (ASTaUND, Kamat et al. 2023), printed as the table used by
loaders/load_surgical_data.py.

The stimulation (tES) studies store the 3D layout of the probe (8 sources, 19 long and 8 short detectors) in an
undocumented frame, in cm although labeled mm. The paper says the probe was placed with respect to the 10-5 system,
so the layout is moved into MNI with the one similarity transform (rotation, shift, uniform scale) that best fits
its sources and long detectors to the nearest 10-5 points of the NIRx ICBM152 template; the nearest points are
updated until they no longer change. The optodes keep their own layout: they are moved together, not snapped onto
10-5 points.

Run from the repository root:  python -m scripts.surgical_positions
"""

import h5py
import numpy as np

from loaders.utils import DATA_ROOT

STIM_FILE = next((DATA_ROOT / "surgical/Long-Term_tES").glob("sub-*/ses-*/nirs/*.snirf"))  # every tES file has the same probe
TEMPLATE_FILE = next((DATA_ROOT / "appearance_dataset").glob("*/*.snirf"))                 # holds the NIRx ICBM152 10-5 table


def similarity_fit(src, dst):
    """Rotation R, uniform scale s and shift t that best map src onto dst (least squares, Umeyama 1991)."""
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    u, sig, vt = np.linalg.svd((dst - dst_mean).T @ (src - src_mean) / len(src))
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])           # no reflection
    rotation = u @ d @ vt
    scale = np.trace(np.diag(sig) @ d) / (src - src_mean).var(0).sum()
    return rotation, scale, dst_mean - scale * rotation @ src_mean


with h5py.File(STIM_FILE, "r") as f:
    sources = f["nirs/probe/sourcePos3D"][()] * 10                # cm -> mm
    detectors = f["nirs/probe/detectorPos3D"][()] * 10
names = [f"S{i + 1}" for i in range(len(sources))] + [f"D{i + 1}" for i in range(len(detectors))]
layout = np.vstack([sources, detectors])                         # 8 sources, then D1-D19 long and D20-D27 short
fit_points = layout[:8 + 19]                                     # sources and long detectors (short ones are 8 mm off)

with h5py.File(TEMPLATE_FILE, "r") as f:
    labels = [label.decode() for label in f["nirs/probe/landmarkLabels"][()]]
    template = f["nirs/probe/landmarkPos3D"][()][:, :3]          # NIRx ICBM152 10-5 positions, MNI mm

rotation, scale, shift = np.eye(3), 1.0, np.zeros(3)
nearest = None
while True:
    moved = scale * fit_points @ rotation.T + shift
    new_nearest = np.linalg.norm(moved[:, None] - template[None], axis=2).argmin(1)  # nearest 10-5 point of each optode
    if nearest is not None and np.array_equal(new_nearest, nearest):
        break
    nearest = new_nearest
    rotation, scale, shift = similarity_fit(fit_points, template[nearest])

mni = scale * layout @ rotation.T + shift
residual = np.linalg.norm(mni[:8 + 19] - template[nearest], axis=1)
angle = np.degrees(np.arccos((np.trace(rotation) - 1) / 2))
print(f"scale {scale:.3f}, rotation {angle:.1f} deg, shift {np.round(shift, 1)} mm; "
      f"distance to the nearest 10-5 point: median {np.median(residual):.1f}, max {residual.max():.1f} mm")
for i, (name, xyz) in enumerate(zip(names, mni)):
    note = f"near {labels[nearest[i]]}" if i < 8 + 19 else "short detector"
    print(f'    "{name}": ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}),  # {note}')
