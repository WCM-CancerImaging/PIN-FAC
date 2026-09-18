from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from micc_model import MICC_FACNet


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        # Plain state_dict
        if obj and all(isinstance(k, str) for k in obj.keys()) and any(
            isinstance(v, torch.Tensor) for v in obj.values()
        ):
            tensor_keys = [k for k, v in obj.items() if isinstance(v, torch.Tensor)]
            # Most plain state_dicts contain only tensors.
            if len(tensor_keys) >= max(1, len(obj) // 2):
                return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}

        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]

    raise RuntimeError(
        "Could not find a PyTorch state_dict in the checkpoint. "
        "Expected a plain state_dict or one of: state_dict/model_state_dict/model."
    )


def load_model(
    checkpoint: str | Path,
    device: torch.device,
    *,
    anchor_gate_init: float = -2.0,
) -> MICC_FACNet:
    model = MICC_FACNet(
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        wn_gate_init=0.0,
        anchor_gate_init=float(anchor_gate_init),
    ).to(device)

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    try:
        obj = torch.load(str(checkpoint), map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(str(checkpoint), map_location=device)

    state = _extract_state_dict(obj)

    if any(k.startswith("module.") for k in state):
        state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

    model.load_state_dict(state, strict=True)
    model.eval()
    return model
