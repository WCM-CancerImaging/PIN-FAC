from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def _valid_values(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = (mask > 0) & np.isfinite(img)
    return img[valid]


def _dynamic_range(img: np.ndarray, mask: np.ndarray, lo=1.0, hi=99.0):
    values = _valid_values(img, mask)
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(values, lo))
    vmax = float(np.percentile(values, hi))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def save_preview(
    echo1: np.ndarray,
    maps: Dict[str, np.ndarray],
    mask2d: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Layout:
        Echo-1 | FF   | R2*  | Ndb
        Nmidb  | SFA  | MUFA | PUFA
    """
    panels = [
        ("Echo-1", echo1, "gray", None),
        ("FF", maps["FF"], "gray", (0.0, 1.0)),
        ("R2*", maps["R2"], "gray", "dynamic"),
        ("Ndb", maps["ndbb"], "gray", "dynamic"),
        ("Nmidb", maps["nmidbb"], "gray", "dynamic"),
        ("SFA", maps["SFA"], "gray", (0.0, 1.0)),
        ("MUFA", maps["MUFA"], "gray", (0.0, 1.0)),
        ("PUFA", maps["PUFA"], "gray", (0.0, 1.0)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12.8, 7.2))
    axes = axes.ravel()

    for ax, (title, img_in, cmap_name, scale) in zip(axes, panels):
        img = np.asarray(img_in, dtype=np.float32).copy()

        if title == "Echo-1":
            img[mask2d == 0] = 0.0
            valid = img[(mask2d > 0) & np.isfinite(img) & (img > 0)]
            vmin = 0.0
            vmax = float(np.percentile(valid, 99.5)) if valid.size else 1.0
        else:
            img[mask2d == 0] = np.nan
            if scale == "dynamic":
                vmin, vmax = _dynamic_range(img, mask2d)
            else:
                vmin, vmax = scale

        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("black")

        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.06)
        cbar.set_ticks([vmin, vmax])

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
