from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .canonical import ActorState, ActorType, SceneState


@dataclass(frozen=True)
class ExplicitActorMatrixSchema:
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

    @property
    def mapped_columns(self) -> tuple[int, ...]:
        values = [self.x, self.y, self.heading, self.length, self.width, self.actor_type,
                  self.track_id, self.valid, self.speed, self.vx, self.vy]
        return tuple(sorted({int(v) for v in values if v is not None}))


class ExplicitMatrixSceneAdapter:
    def __init__(self, schema: ExplicitActorMatrixSchema, actor_type_map: Mapping[int, ActorType | str], *, frame: str) -> None:
        self.schema = schema
        self.actor_type_map = {int(k): (v if isinstance(v, ActorType) else ActorType(str(v))) for k, v in actor_type_map.items()}
        self.inverse_actor_type_map = {v: k for k, v in self.actor_type_map.items()}
        self.frame = str(frame)

    def to_canonical(self, legacy_scene: Any, *, scene_id: str) -> SceneState:
        matrix = np.asarray(legacy_scene)
        if matrix.ndim != 2:
            raise ValueError(f"legacy actor matrix must be 2-D, got {matrix.shape}")
        actors = []
        for i, row in enumerate(matrix):
            heading = float(row[self.schema.heading])
            if self.schema.vx is not None:
                velocity = np.array([float(row[self.schema.vx]), float(row[self.schema.vy])])
            else:
                speed = float(row[self.schema.speed])
                velocity = np.array([speed * np.cos(heading), speed * np.sin(heading)])
            type_code = int(round(float(row[self.schema.actor_type])))
            track_id = str(i) if self.schema.track_id is None else str(int(float(row[self.schema.track_id])))
            valid = True if self.schema.valid is None else bool(row[self.schema.valid])
            actors.append(ActorState(
                track_id=track_id,
                actor_type=self.actor_type_map.get(type_code, ActorType.UNKNOWN),
                position_xy=[row[self.schema.x], row[self.schema.y]],
                heading_rad=heading,
                velocity_xy=velocity,
                length_m=row[self.schema.length],
                width_m=row[self.schema.width],
                valid=valid,
                source_index=i,
            ))
        return SceneState(scene_id=str(scene_id), actors=actors, frame=self.frame)

    def from_canonical(self, scene: SceneState, *, template: Any) -> np.ndarray:
        matrix = np.asarray(template).copy()
        for i, actor in enumerate(scene.actors):
            idx = actor.source_index if actor.source_index is not None else i
            row = matrix[idx]
            row[self.schema.x] = actor.x
            row[self.schema.y] = actor.y
            row[self.schema.heading] = actor.heading_rad
            row[self.schema.length] = actor.length_m
            row[self.schema.width] = actor.width_m
            row[self.schema.actor_type] = self.inverse_actor_type_map[actor.actor_type]
            if self.schema.track_id is not None:
                row[self.schema.track_id] = float(actor.track_id)
            if self.schema.valid is not None:
                row[self.schema.valid] = 1 if actor.valid else 0
            if self.schema.speed is not None:
                row[self.schema.speed] = actor.speed_mps
            if self.schema.vx is not None:
                row[self.schema.vx] = actor.vx
                row[self.schema.vy] = actor.vy
        return matrix
