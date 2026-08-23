#!/usr/bin/env python3
"""Verify every Python import used by the deployment package."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


MODULES = (
    "torch",
    "torchvision",
    "onnx",
    "onnxruntime",
    "yaml",
    "numpy",
    "cv2",
    "nuplan",
    "hydra",
    "pytorch_lightning",
    "diffusers",
    "omegaconf",
    "tensorrt",
)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    results = {}
    failures = []
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            results[name] = {"ok": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as exc:  # import-time binary/linker errors are relevant here
            results[name] = {"ok": False, "error": repr(exc)}
            failures.append(name)

    deployment_imports = (
        "deployment.sledge_rvae.python.contract",
        "deployment.sledge_rvae.python.model",
        "deployment.sledge_rvae.python.comparison",
    )
    for name in deployment_imports:
        try:
            importlib.import_module(name)
            results[name] = {"ok": True}
        except Exception as exc:
            results[name] = {"ok": False, "error": repr(exc)}
            failures.append(name)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "imports": results,
        "passed": not failures,
        "external_target_dependencies": {
            "cuda_driver_and_gpu": "required for Builder/Engine/performance; unavailable on this host",
            "tensorrt_cpp": "8.6.1 headers/library/trtexec required on target machine",
        },
        "installed_distributions": {
            name: importlib.metadata.version(name)
            for name in (
                "cmake",
                "onnx",
                "onnxruntime",
                "tensorrt",
                "tensorrt-bindings",
                "tensorrt-libs",
                "nvidia-cuda-runtime-cu12",
            )
        },
    }
    output = root / "deployment/sledge_rvae/reports/python_environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(f"Import verification failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
