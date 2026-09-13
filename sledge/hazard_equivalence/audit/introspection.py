from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


def structure_report(obj: Any, *, max_depth: int = 4, max_items: int = 8) -> dict[str, Any]:
    """Create a JSON-serializable structural report without assuming SLEDGE field names.

    Call this *inside* the existing dataset/feature/RVAE/simulation code at the exact
    interface you want to audit. It intentionally does not unpickle arbitrary files.
    """

    def walk(value: Any, depth: int) -> Any:
        if depth > max_depth:
            return {"type": _typename(value), "truncated": True}

        if isinstance(value, np.ndarray):
            return {
                "type": "numpy.ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite_fraction": _finite_fraction(value),
                "sample": _sample_array(value, max_items),
            }

        # Torch is optional: detect by interface, do not import it as a hard dependency.
        if _is_torch_tensor(value):
            detached = value.detach().cpu()
            report = {
                "type": _typename(value),
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "device": str(value.device),
            }
            try:
                report["sample"] = detached.reshape(-1)[:max_items].tolist()
            except Exception:
                pass
            return report

        if is_dataclass(value) and not isinstance(value, type):
            return {
                "type": _typename(value),
                "fields": {
                    f.name: walk(getattr(value, f.name), depth + 1)
                    for f in fields(value)
                },
            }

        if isinstance(value, dict):
            items = list(value.items())[:max_items]
            return {
                "type": _typename(value),
                "len": len(value),
                "items": {str(k): walk(v, depth + 1) for k, v in items},
                "truncated": len(value) > max_items,
            }

        if isinstance(value, (list, tuple)):
            return {
                "type": _typename(value),
                "len": len(value),
                "items": [walk(v, depth + 1) for v in value[:max_items]],
                "truncated": len(value) > max_items,
            }

        if isinstance(value, (str, int, float, bool)) or value is None:
            return {"type": _typename(value), "value": value}

        attrs = {}
        if hasattr(value, "__dict__"):
            for k, v in list(vars(value).items())[:max_items]:
                if not k.startswith("_"):
                    attrs[k] = walk(v, depth + 1)
        return {"type": _typename(value), "attrs": attrs}

    return walk(obj, 0)


def _typename(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_torch_tensor(obj: Any) -> bool:
    cls = type(obj)
    return cls.__module__.startswith("torch") and hasattr(obj, "detach") and hasattr(obj, "shape")


def _finite_fraction(array: np.ndarray) -> float | None:
    if not np.issubdtype(array.dtype, np.number) or array.size == 0:
        return None
    return float(np.isfinite(array).mean())


def _sample_array(array: np.ndarray, max_items: int) -> list[Any]:
    flat = array.reshape(-1)
    values = flat[:max_items]
    out: list[Any] = []
    for value in values:
        if isinstance(value, np.generic):
            out.append(value.item())
        else:
            out.append(value)
    return out
