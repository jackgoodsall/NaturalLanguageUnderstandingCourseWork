#!/usr/bin/env python3
"""Probe the active Python environment used for Solution C runs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import site
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class ImportTimeoutError(RuntimeError):
    pass


def alarm_handler(signum: int, frame: Any) -> None:
    raise ImportTimeoutError(f"Timed out while importing module after signal {signum}")


def timed_import(module_name: str, timeout_seconds: int) -> dict[str, Any]:
    started = time.time()
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout_seconds)
    try:
        module = importlib.import_module(module_name)
        signal.alarm(0)
        return {
            "module": module_name,
            "status": "ok",
            "seconds": round(time.time() - started, 4),
            "file": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:  # noqa: BLE001
        signal.alarm(0)
        return {
            "module": module_name,
            "status": "error",
            "seconds": round(time.time() - started, 4),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def torch_details() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}

    details: dict[str, Any] = {
        "status": "ok",
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        details["cuda_device_count"] = torch.cuda.device_count()
        details["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return details


def kernelspecs() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["jupyter", "kernelspec", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}


def build_summary(import_timeout_seconds: int) -> dict[str, Any]:
    return {
        "sys_executable": sys.executable,
        "sys_version": sys.version,
        "cwd": os.getcwd(),
        "path_head": sys.path[:10],
        "site_packages": site.getsitepackages(),
        "usersite": site.getusersitepackages(),
        "env": {
            "PATH": os.environ.get("PATH"),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
        "imports": [
            timed_import(name, import_timeout_seconds)
            for name in ["torch", "transformers", "datasets", "numpy", "pandas", "sklearn"]
        ],
        "torch": torch_details(),
        "kernelspecs": kernelspecs(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--import-timeout-seconds",
        type=int,
        default=20,
        help="Seconds to allow per module import before marking it as hung",
    )
    parser.add_argument("--json-out", help="Optional path to write JSON output")
    args = parser.parse_args()

    summary = build_summary(args.import_timeout_seconds)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload, flush=True)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
