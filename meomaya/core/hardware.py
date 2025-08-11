from __future__ import annotations

import os
from typing import Optional


def select_device(prefer_gpu: bool = True) -> str:
    """
    Select an available device string among 'cuda', 'mps', or 'cpu'.

    - Honors environment variable MEOMAYA_DEVICE if set (e.g., 'cpu', 'cuda', 'mps')
    - Avoids hard dependency on torch; falls back to 'cpu' if torch is missing
    - Supports Apple Silicon via MPS when available (PyTorch compiled with MPS)
    """
    env_device = (os.getenv("MEOMAYA_DEVICE") or "").lower().strip()
    if env_device in {"cpu", "cuda", "mps"}:
        return env_device

    if not prefer_gpu:
        return "cpu"

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon Metal Performance Shaders
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        # torch not installed or not configured; default to CPU
        pass
    return "cpu"


__all__ = ["select_device"]


