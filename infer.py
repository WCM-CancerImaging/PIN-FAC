from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from facnet.model_loader import load_model
from facnet.postprocess import masked, scale_model_outputs
from facnet.preprocess import prepare_single_slice
from facnet.visualize import save_preview


DEFAULT_SAVE_KEYS = ["FF", "SFA", "MUFA", "PUFA", "R2", "ndbb", "nmidbb"]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return torch.device(name)


@torch.no_grad()
def infer_voxels(
    prepared,
    model,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    mag = prepared["mag"]
    coords = prepared["voxel_idx"]
    x_reim = prepared["x_reim"]
    anchor_log = prepared["anchor_log"]
    te_norm_np = prepared["te_norm"]

    H, W, NE = mag.shape
    outmap = np.zeros((H, W, 9), dtype=np.float32)

    te_norm = torch.from_numpy(te_norm_np).float().view(1, NE, 1).to(device)

    for start in range(0, len(coords), int(batch_size)):
        chunk = coords[start : start + int(batch_size)]
        y = chunk[:, 0].astype(np.int64)
        x = chunk[:, 1].astype(np.int64)

        x_val = torch.from_numpy(
            np.array(x_reim[y, x, :, :], copy=True)
        ).float().to(device)

        anchor = torch.from_numpy(
            np.array(anchor_log[y, x], copy=True)
        ).float().view(-1, 1).to(device)

        te_batch = te_norm.expand(x_val.shape[0], -1, -1)

        pred = model(x_val, te_batch, anchor)
        pred = pred.squeeze(-1).squeeze(-1).cpu().numpy()
        outmap[y, x, :] = pred

    return outmap


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "FAC-Net inference for one multi-echo slice stored as "
            "Image/*.dcm and Phase/*.dcm."
        )
    )
    parser.add_argument("--image-dir", required=True, help="Magnitude DICOM folder.")
    parser.add_argument("--phase-dir", required=True, help="Phase DICOM folder.")
    parser.add_argument("--weights", required=True, help="Pretrained model checkpoint.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--expected-echoes", type=int, default=16)
    parser.add_argument("--voxel-eps", type=float, default=1e-6)
    parser.add_argument("--m0-threshold", type=float, default=50.0)
    parser.add_argument("--mask-echo-idx", type=int, default=1)
    parser.add_argument("--mask-rel-to-m0", type=float, default=0.08)
    parser.add_argument("--anchor-gate-init", type=float, default=-2.0)
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Also save secondary model-derived outputs.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_single_slice(
        image_dir=args.image_dir,
        phase_dir=args.phase_dir,
        expected_echoes=args.expected_echoes,
        voxel_eps=args.voxel_eps,
        m0_threshold=args.m0_threshold,
        mask_echo_idx=args.mask_echo_idx,
        mask_rel_to_m0=args.mask_rel_to_m0,
    )

    if prepared["voxel_idx"].shape[0] == 0:
        raise RuntimeError(
            "The preprocessing mask contains zero voxels. "
            "Check the DICOM input and mask thresholds."
        )

    model = load_model(
        args.weights,
        device=device,
        anchor_gate_init=args.anchor_gate_init,
    )

    outmap = infer_voxels(
        prepared=prepared,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    maps = scale_model_outputs(outmap, prepared["m0_raw"])
    mask2d = prepared["mask2d"]

    save_keys = list(DEFAULT_SAVE_KEYS)
    if args.save_all:
        save_keys = list(maps.keys())

    for key in save_keys:
        np.save(out_dir / f"{key}.npy", masked(maps[key], mask2d))

    np.save(out_dir / "mask2d.npy", mask2d.astype(np.uint8))
    np.save(out_dir / "te_ms.npy", prepared["te_ms"].astype(np.float32))

    save_preview(
        echo1=prepared["mag"][..., 0],
        maps=maps,
        mask2d=mask2d,
        output_path=out_dir / "preview.png",
    )

    manifest = {
        "input": {
            "image_dir": str(Path(args.image_dir)),
            "phase_dir": str(Path(args.phase_dir)),
            "n_echoes": int(prepared["mag"].shape[-1]),
            "te_ms": [float(x) for x in prepared["te_ms"]],
            "image_sort_method": prepared["image_sort_method"],
            "phase_sort_method": prepared["phase_sort_method"],
            "image_files_sorted_by_echo_time": [
                Path(x).name for x in prepared["image_files"].tolist()
            ],
            "phase_files_sorted_by_echo_time": [
                Path(x).name for x in prepared["phase_files"].tolist()
            ],
        },
        "preprocessing": {
            "voxel_eps": float(args.voxel_eps),
            "m0_threshold": float(args.m0_threshold),
            "mask_echo_idx": int(args.mask_echo_idx),
            "mask_rel_to_m0": float(args.mask_rel_to_m0),
            "voxel_count": int(prepared["voxel_idx"].shape[0]),
        },
        "model": {
            "weights": str(Path(args.weights)),
            "device": str(device),
        },
        "saved_maps": save_keys,
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Done. Results saved to: {out_dir}")
    print(f"Valid voxels: {prepared['voxel_idx'].shape[0]}")
    print(f"Echo times (ms): {prepared['te_ms'].tolist()}")


if __name__ == "__main__":
    main()
