from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..adapters.base import SceneAdapter
from ..core.types import ActorState, SceneState
from ..core.validation import validate_scene


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
    scene_validation_errors: list[str]
    messages: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _angle_error(a: float, b: float) -> float:
    return abs(float(np.arctan2(np.sin(a - b), np.cos(a - b))))


def compare_canonical_scenes(
    a: SceneState,
    b: SceneState,
    *,
    atol_position_m: float = 1e-6,
    atol_heading_rad: float = 1e-6,
    atol_velocity_mps: float = 1e-6,
    atol_size_m: float = 1e-6,
) -> RoundTripReport:
    messages: list[str] = []
    validation_errors = validate_scene(a) + validate_scene(b)
    if len(a.actors) != len(b.actors):
        messages.append(f"actor count mismatch: {len(a.actors)} != {len(b.actors)}")
        return RoundTripReport(
            ok=False,
            max_position_error_m=float("inf"),
            max_heading_error_rad=float("inf"),
            max_velocity_error_mps=float("inf"),
            max_size_error_m=float("inf"),
            actor_type_mismatches=abs(len(a.actors) - len(b.actors)),
            track_id_mismatches=abs(len(a.actors) - len(b.actors)),
            valid_mismatches=abs(len(a.actors) - len(b.actors)),
            scene_validation_errors=validation_errors,
            messages=messages,
        )

    max_pos = max_heading = max_vel = max_size = 0.0
    type_mm = id_mm = valid_mm = 0

    for i, (aa, bb) in enumerate(zip(a.actors, b.actors)):
        pos = float(np.linalg.norm(aa.position_xy - bb.position_xy))
        vel = float(np.linalg.norm(aa.velocity_xy - bb.velocity_xy))
        heading = _angle_error(aa.heading_rad, bb.heading_rad)
        size = max(abs(aa.length_m - bb.length_m), abs(aa.width_m - bb.width_m))
        max_pos = max(max_pos, pos)
        max_vel = max(max_vel, vel)
        max_heading = max(max_heading, heading)
        max_size = max(max_size, size)
        type_mm += int(aa.actor_type != bb.actor_type)
        id_mm += int(aa.track_id != bb.track_id)
        valid_mm += int(aa.valid != bb.valid)
        if pos > atol_position_m:
            messages.append(f"actor[{i}] position error={pos:.6g} m")
        if heading > atol_heading_rad:
            messages.append(f"actor[{i}] heading error={heading:.6g} rad")
        if vel > atol_velocity_mps:
            messages.append(f"actor[{i}] velocity error={vel:.6g} m/s")
        if size > atol_size_m:
            messages.append(f"actor[{i}] size error={size:.6g} m")

    ok = (
        not validation_errors
        and max_pos <= atol_position_m
        and max_heading <= atol_heading_rad
        and max_vel <= atol_velocity_mps
        and max_size <= atol_size_m
        and type_mm == 0
        and id_mm == 0
        and valid_mm == 0
    )
    return RoundTripReport(
        ok=ok,
        max_position_error_m=max_pos,
        max_heading_error_rad=max_heading,
        max_velocity_error_mps=max_vel,
        max_size_error_m=max_size,
        actor_type_mismatches=type_mm,
        track_id_mismatches=id_mm,
        valid_mismatches=valid_mm,
        scene_validation_errors=validation_errors,
        messages=messages,
    )


def roundtrip_check(
    adapter: SceneAdapter,
    legacy_scene: Any,
    *,
    scene_id: str,
) -> tuple[SceneState, Any, SceneState, RoundTripReport]:
    canonical_before = adapter.to_canonical(legacy_scene, scene_id=scene_id)
    legacy_after = adapter.from_canonical(canonical_before, template=legacy_scene)
    canonical_after = adapter.to_canonical(legacy_after, scene_id=scene_id)
    report = compare_canonical_scenes(canonical_before, canonical_after)
    return canonical_before, legacy_after, canonical_after, report
