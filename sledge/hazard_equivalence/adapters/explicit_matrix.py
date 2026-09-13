from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..core.types import ActorState, ActorType, SceneState


@dataclass(frozen=True)
class ExplicitActorMatrixSchema:
    """Explicit legacy actor-matrix schema.

    No defaults for semantic columns are provided on purpose: Phase 0 is precisely the
    stage where the real repository definition must be verified instead of guessed.
    """

    x: int
    y: int
    heading: int
    length: int
    width: int
    actor_type: int
    track_id: int | None = None
    valid: int | None = None
    speed: int | None = None
    vx: int | None = None
    vy: int | None = None

    def __post_init__(self) -> None:
        if self.speed is None and (self.vx is None or self.vy is None):
            raise ValueError("Provide either speed or both vx and vy")
        if (self.vx is None) != (self.vy is None):
            raise ValueError("vx and vy must either both be set or both be None")

    @property
    def mapped_columns(self) -> tuple[int, ...]:
        values = [
            self.x,
            self.y,
            self.heading,
            self.length,
            self.width,
            self.actor_type,
            self.track_id,
            self.valid,
            self.speed,
            self.vx,
            self.vy,
        ]
        return tuple(sorted({int(v) for v in values if v is not None}))


class ExplicitMatrixSceneAdapter:
    """Canonical adapter for a legacy [num_actors, num_features] matrix.

    The caller supplies the exact schema and actor-type coding discovered from the
    repository. Unmapped columns are preserved by copying the provided template during
    reverse conversion.
    """

    def __init__(
        self,
        schema: ExplicitActorMatrixSchema,
        actor_type_map: Mapping[int, ActorType | str],
        *,
        frame: str,
    ) -> None:
        self.schema = schema
        self.actor_type_map = {
            int(k): (v if isinstance(v, ActorType) else ActorType(str(v)))
            for k, v in actor_type_map.items()
        }
        self.inverse_actor_type_map: dict[ActorType, int] = {}
        for code, actor_type in self.actor_type_map.items():
            if actor_type in self.inverse_actor_type_map:
                raise ValueError(f"Actor type {actor_type} has multiple legacy codes")
            self.inverse_actor_type_map[actor_type] = code
        self.frame = str(frame)

    def to_canonical(self, legacy_scene: Any, *, scene_id: str) -> SceneState:
        matrix = np.asarray(legacy_scene)
        if matrix.ndim != 2:
            raise ValueError(f"legacy actor matrix must be 2-D, got {matrix.shape}")
        if matrix.shape[1] <= max(self.schema.mapped_columns):
            raise ValueError(
                f"matrix has {matrix.shape[1]} columns, but schema references "
                f"column {max(self.schema.mapped_columns)}"
            )

        actors: list[ActorState] = []
        for row_idx, row in enumerate(matrix):
            valid = True if self.schema.valid is None else bool(row[self.schema.valid])
            type_code = int(round(float(row[self.schema.actor_type])))
            actor_type = self.actor_type_map.get(type_code, ActorType.UNKNOWN)
            track_id = (
                str(row_idx)
                if self.schema.track_id is None
                else _stable_id(row[self.schema.track_id])
            )

            heading = float(row[self.schema.heading])
            if self.schema.vx is not None:
                velocity = np.array(
                    [float(row[self.schema.vx]), float(row[self.schema.vy])],
                    dtype=np.float64,
                )
            else:
                speed = float(row[self.schema.speed])  # type: ignore[index]
                velocity = np.array(
                    [speed * np.cos(heading), speed * np.sin(heading)],
                    dtype=np.float64,
                )

            actors.append(
                ActorState(
                    track_id=track_id,
                    actor_type=actor_type,
                    position_xy=np.array(
                        [float(row[self.schema.x]), float(row[self.schema.y])],
                        dtype=np.float64,
                    ),
                    heading_rad=heading,
                    velocity_xy=velocity,
                    length_m=float(row[self.schema.length]),
                    width_m=float(row[self.schema.width]),
                    valid=valid,
                    source_index=row_idx,
                    metadata={"legacy_actor_type_code": type_code},
                )
            )

        return SceneState(scene_id=str(scene_id), actors=actors, frame=self.frame)

    def from_canonical(self, scene: SceneState, *, template: Any) -> np.ndarray:
        matrix = np.asarray(template).copy()
        if matrix.ndim != 2:
            raise ValueError(f"template actor matrix must be 2-D, got {matrix.shape}")
        if len(scene.actors) != matrix.shape[0]:
            raise ValueError(
                "Phase-0 roundtrip requires identical actor count/order; "
                f"canonical={len(scene.actors)}, template={matrix.shape[0]}"
            )

        for fallback_idx, actor in enumerate(scene.actors):
            row_idx = actor.source_index if actor.source_index is not None else fallback_idx
            if not 0 <= row_idx < matrix.shape[0]:
                raise IndexError(f"source_index out of range: {row_idx}")
            row = matrix[row_idx]

            row[self.schema.x] = actor.x
            row[self.schema.y] = actor.y
            row[self.schema.heading] = actor.heading_rad
            row[self.schema.length] = actor.length_m
            row[self.schema.width] = actor.width_m

            if actor.actor_type not in self.inverse_actor_type_map:
                raise KeyError(f"No legacy code for actor type {actor.actor_type}")
            row[self.schema.actor_type] = self.inverse_actor_type_map[actor.actor_type]

            if self.schema.track_id is not None:
                try:
                    row[self.schema.track_id] = float(actor.track_id)
                except ValueError as exc:
                    raise ValueError(
                        "Legacy track_id column is numeric but canonical track_id is not. "
                        "Use a repository-specific adapter if IDs are encoded differently."
                    ) from exc
            if self.schema.valid is not None:
                row[self.schema.valid] = 1 if actor.valid else 0
            if self.schema.speed is not None:
                row[self.schema.speed] = actor.speed_mps
            if self.schema.vx is not None:
                row[self.schema.vx] = actor.vx
                row[self.schema.vy] = actor.vy

        return matrix


def _stable_id(value: Any) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(value_float) and value_float.is_integer():
        return str(int(value_float))
    return repr(value_float)
