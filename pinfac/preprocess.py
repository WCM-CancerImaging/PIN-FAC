from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pydicom


def _dicom_files(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".dcm"
    )
    if not files:
        raise RuntimeError(f"No .dcm files found in: {folder}")
    return files


def _read_echo_time_ms(ds: pydicom.Dataset, path: Path) -> float:
    if not hasattr(ds, "EchoTime"):
        raise RuntimeError(f"EchoTime is missing: {path}")
    try:
        te = float(ds.EchoTime)
    except Exception as exc:
        raise RuntimeError(f"Invalid EchoTime in {path}") from exc
    if not np.isfinite(te):
        raise RuntimeError(f"Non-finite EchoTime in {path}")
    return te


def _apply_rescale_if_present(ds: pydicom.Dataset, arr: np.ndarray) -> np.ndarray:
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept


def _is_radian_like(arr: np.ndarray) -> bool:
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    return (-3.7 <= vmin <= 3.7) and (-3.7 <= vmax <= 3.7)


def map_phase_to_rad(raw: np.ndarray) -> np.ndarray:
    """
    Match the phase conversion used by the original preprocessing pipeline.

    1) Already-radian data are kept as-is.
    2) Siemens-like 12-bit phase [0, 4095] is mapped to approximately [-pi, pi].
    3) Otherwise values are centered on the median and wrapped to [-pi, pi].
    """
    raw = raw.astype(np.float32)

    if _is_radian_like(raw):
        return raw

    vmin = float(raw.min())
    vmax = float(raw.max())

    if 0.0 <= vmin and vmax <= 4095.0:
        return ((raw - 2048.0) * (math.pi / 2048.0)).astype(np.float32)

    x = raw - np.median(raw)
    x = ((x + math.pi) % (2.0 * math.pi)) - math.pi
    return x.astype(np.float32)


def te_norm_from_te_ms(te_ms: np.ndarray) -> np.ndarray:
    te_s = te_ms.astype(np.float32) / 1000.0
    tmin = float(te_s.min())
    tmax = float(te_s.max())
    if (tmax - tmin) < 1e-12:
        return np.zeros_like(te_s, dtype=np.float32)
    return ((te_s - tmin) / (tmax - tmin)).astype(np.float32)


def _natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _optional_float(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _load_echo_series(
    folder: str | Path,
    *,
    is_phase: bool,
):
    """
    Load one single-slice multi-echo DICOM series.

    Magnitude:
      EchoTime is required and is used for ordering.

    Phase:
      EchoTime is preferred. If it is unavailable in the phase export,
      EchoNumbers/EchoNumber, then InstanceNumber, then natural filename
      order are used as compatibility fallbacks.
    """
    records = []

    for path in _dicom_files(folder):
        ds = pydicom.dcmread(str(path), stop_before_pixels=False)
        arr = ds.pixel_array.astype(np.float32)

        if arr.ndim != 2:
            raise RuntimeError(
                f"Expected one 2-D DICOM frame per file, but got shape {arr.shape}: {path}"
            )

        te_ms = _optional_float(getattr(ds, "EchoTime", None))
        echo_no = _optional_float(
            getattr(ds, "EchoNumbers", getattr(ds, "EchoNumber", None))
        )
        instance_no = _optional_float(getattr(ds, "InstanceNumber", None))

        if (not is_phase) and te_ms is None:
            raise RuntimeError(f"EchoTime is missing or invalid: {path}")

        if is_phase:
            arr = map_phase_to_rad(arr)
        else:
            arr = _apply_rescale_if_present(ds, arr).astype(np.float32)

        records.append(
            {
                "te_ms": te_ms,
                "echo_no": echo_no,
                "instance_no": instance_no,
                "path": path,
                "array": arr,
            }
        )

    if not is_phase:
        records.sort(key=lambda r: r["te_ms"])
        sort_method = "EchoTime"
    else:
        if all(r["te_ms"] is not None for r in records):
            records.sort(key=lambda r: r["te_ms"])
            sort_method = "EchoTime"
        elif all(r["echo_no"] is not None for r in records):
            records.sort(key=lambda r: r["echo_no"])
            sort_method = "EchoNumbers"
        elif all(r["instance_no"] is not None for r in records):
            records.sort(key=lambda r: r["instance_no"])
            sort_method = "InstanceNumber"
        else:
            records.sort(key=lambda r: _natural_key(r["path"]))
            sort_method = "natural filename"

    return records, sort_method

def prepare_single_slice(
    image_dir: str | Path,
    phase_dir: str | Path,
    *,
    expected_echoes: int = 16,
    voxel_eps: float = 1e-6,
    m0_threshold: float = 50.0,
    mask_echo_idx: int = 1,
    mask_rel_to_m0: float = 0.08,
    te_match_tolerance_ms: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """
    Convert:
        Image/*.dcm
        Phase/*.dcm

    into the exact voxel-normalized inputs used by the inference model.
    The sample represents one spatial slice with one DICOM file per echo.
    """
    mag_records, mag_sort_method = _load_echo_series(image_dir, is_phase=False)
    pha_records, pha_sort_method = _load_echo_series(phase_dir, is_phase=True)

    if len(mag_records) != expected_echoes:
        raise RuntimeError(
            f"Expected {expected_echoes} magnitude echoes, found {len(mag_records)} "
            f"in {image_dir}"
        )
    if len(pha_records) != expected_echoes:
        raise RuntimeError(
            f"Expected {expected_echoes} phase echoes, found {len(pha_records)} "
            f"in {phase_dir}"
        )

    mag_te = np.array([r["te_ms"] for r in mag_records], dtype=np.float32)
    phase_has_te = all(r["te_ms"] is not None for r in pha_records)
    if phase_has_te:
        pha_te = np.array([r["te_ms"] for r in pha_records], dtype=np.float32)
        if not np.allclose(mag_te, pha_te, atol=te_match_tolerance_ms, rtol=0.0):
            raise RuntimeError(
                "Magnitude and phase EchoTime values do not match.\n"
                f"Magnitude TE (ms): {mag_te.tolist()}\n"
                f"Phase TE (ms): {pha_te.tolist()}"
            )

    if np.any(np.diff(mag_te) <= 0):
        raise RuntimeError(
            "EchoTime values must be unique and strictly increasing after sorting. "
            f"Found: {mag_te.tolist()}"
        )

    shapes = {r["array"].shape for r in mag_records + pha_records}
    if len(shapes) != 1:
        raise RuntimeError(f"DICOM matrix-size mismatch: {sorted(shapes)}")

    mag = np.stack([r["array"] for r in mag_records], axis=-1).astype(np.float32)
    pha = np.stack([r["array"] for r in pha_records], axis=-1).astype(np.float32)

    H, W, NE = mag.shape
    if not (0 <= mask_echo_idx < NE):
        raise ValueError(f"mask_echo_idx={mask_echo_idx} is outside [0, {NE - 1}]")

    m0_raw = mag[..., 0].astype(np.float32)
    denom = m0_raw + float(voxel_eps)

    mag_norm = (mag / denom[..., None]).astype(np.float32)
    x_re = mag_norm * np.cos(pha)
    x_im = mag_norm * np.sin(pha)
    x_reim = np.stack([x_re, x_im], axis=-1).astype(np.float32)

    anchor_log = np.log(np.clip(m0_raw, float(voxel_eps), None)).astype(np.float32)

    valid_m0 = m0_raw > float(m0_threshold)
    mk = mag[..., int(mask_echo_idx)]
    mask2d = valid_m0 & (mk > float(mask_rel_to_m0) * m0_raw)

    ys, xs = np.where(mask2d)
    voxel_idx = np.stack([ys, xs], axis=1).astype(np.int32)

    return {
        "mag": mag,
        "pha": pha,
        "te_ms": mag_te,
        "te_norm": te_norm_from_te_ms(mag_te),
        "x_reim": x_reim,
        "anchor_log": anchor_log,
        "m0_raw": m0_raw,
        "mask2d": mask2d.astype(np.uint8),
        "voxel_idx": voxel_idx,
        "image_files": np.array([str(r["path"]) for r in mag_records], dtype=object),
        "phase_files": np.array([str(r["path"]) for r in pha_records], dtype=object),
        "image_sort_method": mag_sort_method,
        "phase_sort_method": pha_sort_method,
    }
