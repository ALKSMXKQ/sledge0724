from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class ActorType(str, Enum):
    EGO = "ego"
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    STATIC = "static"
    UNKNOWN = "unknown"


def _vec2(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {arr.shape}")
    return arr


def _polyline(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"polyline must have shape (N, 2), got {arr.shape}")
    return arr


@dataclass
class ActorState:
    """Canonical single-frame actor state.

    Coordinate convention is deliberately not hard-coded beyond being a common 2-D
    Cartesian frame. The adapter must document whether this is world, ego-local, etc.
    Heading is in radians. Size is full box length/width, not half extents.
    """

    track_id: str
    actor_type: ActorType
    position_xy: np.ndarray
    heading_rad: float
    velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    length_m: float = 0.0
    width_m: float = 0.0
    valid: bool = True
    timestamp_s: float | None = None
    source_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position_xy = _vec2(self.position_xy, "position_xy")
        self.velocity_xy = _vec2(self.velocity_xy, "velocity_xy")
        self.heading_rad = float(self.heading_rad)
        self.length_m = float(self.length_m)
        self.width_m = float(self.width_m)
        if self.timestamp_s is not None:
            self.timestamp_s = float(self.timestamp_s)
        if not isinstance(self.actor_type, ActorType):
            self.actor_type = ActorType(str(self.actor_type))

    @property
    def x(self) -> float:
        return float(self.position_xy[0])

    @property
    def y(self) -> float:
        return float(self.position_xy[1])

    @property
    def vx(self) -> float:
        return float(self.velocity_xy[0])

    @property
    def vy(self) -> float:
        return float(self.velocity_xy[1])

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.velocity_xy))

    @property
    def size_lw(self) -> tuple[float, float]:
        return self.length_m, self.width_m


@dataclass
class LaneState:
    lane_id: str
    centerline_xy: np.ndarray
    successor_ids: tuple[str, ...] = ()
    predecessor_ids: tuple[str, ...] = ()
    is_drivable: bool = True
    speed_limit_mps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.centerline_xy = _polyline(self.centerline_xy)
        self.successor_ids = tuple(map(str, self.successor_ids))
        self.predecessor_ids = tuple(map(str, self.predecessor_ids))
        if self.speed_limit_mps is not None:
            self.speed_limit_mps = float(self.speed_limit_mps)


@dataclass
class SceneState:
    scene_id: str
    actors: list[ActorState]
    lanes: list[LaneState] = field(default_factory=list)
    frame: str = "unspecified"
    timestamp_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def actor_by_id(self, track_id: str) -> ActorState:
        matches = [actor for actor in self.actors if actor.track_id == str(track_id)]
        if len(matches) != 1:
            raise KeyError(f"Expected exactly one actor with track_id={track_id!r}, got {len(matches)}")
        return matches[0]

    @property
    def pedestrians(self) -> list[ActorState]:
        return [a for a in self.actors if a.actor_type == ActorType.PEDESTRIAN and a.valid]

    @property
    def vehicles(self) -> list[ActorState]:
        return [a for a in self.actors if a.actor_type in {ActorType.EGO, ActorType.VEHICLE} and a.valid]


@dataclass
class TrajectoryState:
    """Canonical multi-actor trajectory representation.

    Shapes:
      timestamps_s: [T]
      actor_ids: N strings
      positions_xy: [T, N, 2]
      headings_rad: [T, N]
      valid: [T, N]
    """

    timestamps_s: np.ndarray
    actor_ids: tuple[str, ...]
    positions_xy: np.ndarray
    headings_rad: np.ndarray
    valid: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps_s = np.asarray(self.timestamps_s, dtype=np.float64)
        self.actor_ids = tuple(map(str, self.actor_ids))
        self.positions_xy = np.asarray(self.positions_xy, dtype=np.float64)
        self.headings_rad = np.asarray(self.headings_rad, dtype=np.float64)
        self.valid = np.asarray(self.valid, dtype=bool)
