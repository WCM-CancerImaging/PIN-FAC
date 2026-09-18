from __future__ import annotations

from typing import Dict

import numpy as np


# Exact model-output scaling values used during PIN-FAC training.
W_MEAN_R, W_STD_R = 0.2, 0.1
F_MEAN_R, F_STD_R = 0.1, 0.2
FRQ_MEAN, FRQ_STD = 30.0, 30.0
R2_MEAN, R2_STD = 80.0, 50.0
W_MEAN_I, W_STD_I = -1.0, 1.0
F_MEAN_I, F_STD_I = -1.0, 1.0
P_MEAN, P_STD = 0.1, 0.2
NDB_MEAN, NDB_STD = 0.1, 0.2
NMIDB_MEAN, NMIDB_STD = 0.1, 0.2


def scale_model_outputs(outmap: np.ndarray, m0_raw: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert the 9 normalized network outputs to physical/derived maps.

    Network output channel order:
      0 water_r, 1 fat_r, 2 R2, 3 Ndb, 4 Nmidb,
      5 water_i, 6 fat_i, 7 frequency, 8 p
    """
    if outmap.ndim != 3 or outmap.shape[-1] != 9:
        raise ValueError(f"Expected outmap (H,W,9), got {outmap.shape}")

    w_r, f_r, r2, ndb, nmidb, w_i, f_i, frq, p = [
        outmap[..., i] for i in range(9)
    ]

    water_rel = np.abs(W_STD_R * w_r + W_MEAN_R)
    fat_rel = np.abs(F_STD_R * f_r + F_MEAN_R)

    R2 = R2_STD * r2 + R2_MEAN
    ndbb = np.abs(NDB_STD * ndb + NDB_MEAN)
    nmidbb = np.abs(NMIDB_STD * nmidb + NMIDB_MEAN)

    wat_i = W_STD_I * w_i + W_MEAN_I
    fat_i = F_STD_I * f_i + F_MEAN_I
    frq_hz = FRQ_STD * frq + FRQ_MEAN
    p_scaled = P_STD * p + P_MEAN

    eps = 1e-8

    # Restore water/fat to the raw echo-0 scale, matching the original inference code.
    water = water_rel * m0_raw
    fat = fat_rel * m0_raw
    water_plus_fat = water + fat
    FF = fat / (water_plus_fat + eps)

    # FAC definitions used during training/inference.
    UFA = (ndbb - nmidbb) / 3.0
    PUFA = nmidbb / 3.0
    SFA = 1.0 - UFA
    MUFA = UFA - PUFA

    return {
        "water": water.astype(np.float32),
        "fat": fat.astype(np.float32),
        "FF": FF.astype(np.float32),
        "R2": R2.astype(np.float32),
        "ndbb": ndbb.astype(np.float32),
        "nmidbb": nmidbb.astype(np.float32),
        "SFA": SFA.astype(np.float32),
        "MUFA": MUFA.astype(np.float32),
        "PUFA": PUFA.astype(np.float32),
        "UFA": UFA.astype(np.float32),
        "wat_i": wat_i.astype(np.float32),
        "fat_i": fat_i.astype(np.float32),
        "frq": frq_hz.astype(np.float32),
        "p": p_scaled.astype(np.float32),
        "water_plus_fat": water_plus_fat.astype(np.float32),
    }


def masked(map2d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    return (map2d.astype(np.float32) * (mask2d > 0).astype(np.float32)).astype(np.float32)
