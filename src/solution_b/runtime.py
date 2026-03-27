"""
runtime.py - Runtime helpers for Solution B.

Keeps device selection consistent across training, evaluation, and HPO.
"""

import torch


def mps_available() -> bool:
    """Return True when Apple Metal acceleration is usable."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(requested: str = "auto") -> str:
    """Resolve a requested device name to an available torch device string."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available():
            return "mps"
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but no CUDA device is available")
        return "cuda"

    if requested == "mps":
        if not mps_available():
            raise ValueError("MPS requested but Apple Metal acceleration is unavailable")
        return "mps"

    if requested == "cpu":
        return "cpu"

    raise ValueError(f"Unsupported device request: {requested}")
