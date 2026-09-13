from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

import numpy as np

from .canonical import SceneState


def structure_report(obj: Any, *, max_depth: int = 4, max_items: int = 8) -> dict[str, Any]:
    def walk(value: Any, depth: int) -> Any:
        if depth > max_depth:
            return {"type": _typename(value), "truncated": True}
        if isinstance(value, np.ndarray):
            return {
                "type": "numpy.ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sample": [x.item() if isinstance(x, np.generic) else x for x in value.reshape(-1)[:max_items]],
            }
        if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
            t = value.detach().cpu()
            return {"type": _typename(value), "shape": list(t.shape), "dtype": str(t.dtype), "sample": t.reshape(-1)[:max_items].tolist()}
        if is_dataclass(value) and not isinstance(value, type):
            return {"type": _typename(value), "fields": {f.name: walk(getattr(value, f.name), depth + 1) for f in fields(value)}}
        if isinstance(value, dict):
            items = list(value.items())[:max_items]
            return {"type": _typename(value), "len": len(value), "items": {str(k): walk(v, depth + 1) for k, v in items}}
        if isinstance(value, (list, tuple)):
            return {"type": _typename(value), "len": len(value), "items": [walk(v, depth + 1) for v in value[:max_items]]}
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


@dataclass
class RoundTripReport:
    ok: bool
    max_position_error_m: float
    max_heading_error_rad: float
    max_velocity_error_mps: float
    max_size_error_m: float
    actor_type_mismatches: int
    track_id_mismatches: int
    valid_mismatches: int

    def to_dict(self):
        return asdict(self)


def compare_canonical_scenes(a: SceneState, b: SceneState, tol: float = 1e-6) -> RoundTripReport:
    if len(a.actors) != len(b.actors):
        return RoundTripReport(False, float("inf"), float("inf"), float("inf"), float("inf"), 1, 1, 1)
    max_pos = max_heading = max_vel = max_size = 0.0
    type_mm = id_mm = valid_mm = 0
    for aa, bb in zip(a.actors, b.actors):
        max_pos = max(max_pos, float(np.linalg.norm(aa.position_xy - bb.position_xy)))
        max_vel = max(max_vel, float(np.linalg.norm(aa.velocity_xy - bb.velocity_xy)))
        max_heading = max(max_heading, abs(float(np.arctan2(np.sin(aa.heading_rad - bb.heading_rad), np.cos(aa.heading_rad - bb.heading_rad)))))
        max_size = max(max_size, abs(aa.length_m - bb.length_m), abs(aa.width_m - bb.width_m))
        type_mm += int(aa.actor_type != bb.actor_type)
        id_mm += int(aa.track_id != bb.track_id)
        valid_mm += int(aa.valid != bb.valid)
    ok = max(max_pos, max_heading, max_vel, max_size) <= tol and type_mm == id_mm == valid_mm == 0
    return RoundTripReport(ok, max_pos, max_heading, max_vel, max_size, type_mm, id_mm, valid_mm)


def roundtrip_check(adapter, legacy_scene: Any, *, scene_id: str):
    before = adapter.to_canonical(legacy_scene, scene_id=scene_id)
    legacy_after = adapter.from_canonical(before, template=legacy_scene)
    after = adapter.to_canonical(legacy_after, scene_id=scene_id)
    return before, legacy_after, after, compare_canonical_scenes(before, after)
