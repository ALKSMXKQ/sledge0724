from __future__ import annotations

import math

import numpy as np

from .types import ActorType, SceneState, TrajectoryState


def validate_scene(scene: SceneState) -> list[str]:
    """Return validation errors. Empty list means valid for Phase-0 representation use."""
    errors: list[str] = []
    ids: set[str] = set()

    for i, actor in enumerate(scene.actors):
        prefix = f"actors[{i}]/{actor.track_id}"
        if actor.track_id in ids:
            errors.append(f"{prefix}: duplicate track_id")
        ids.add(actor.track_id)

        if not np.isfinite(actor.position_xy).all():
            errors.append(f"{prefix}: non-finite position")
        if not np.isfinite(actor.velocity_xy).all():
            errors.append(f"{prefix}: non-finite velocity")
        if not math.isfinite(actor.heading_rad):
            errors.append(f"{prefix}: non-finite heading")
        if actor.length_m < 0 or actor.width_m < 0:
            errors.append(f"{prefix}: negative size")
        if actor.valid and actor.actor_type in {
            ActorType.EGO,
            ActorType.VEHICLE,
            ActorType.PEDESTRIAN,
            ActorType.CYCLIST,
            ActorType.STATIC,
        }:
            if actor.length_m <= 0 or actor.width_m <= 0:
                errors.append(f"{prefix}: valid physical actor must have positive length/width")

    lane_ids: set[str] = set()
    for i, lane in enumerate(scene.lanes):
        prefix = f"lanes[{i}]/{lane.lane_id}"
        if lane.lane_id in lane_ids:
            errors.append(f"{prefix}: duplicate lane_id")
        lane_ids.add(lane.lane_id)
        if len(lane.centerline_xy) < 2:
            errors.append(f"{prefix}: centerline needs >=2 points")
        if not np.isfinite(lane.centerline_xy).all():
            errors.append(f"{prefix}: non-finite centerline")

    for lane in scene.lanes:
        for successor in lane.successor_ids:
            if successor not in lane_ids:
                errors.append(f"lane/{lane.lane_id}: unknown successor {successor}")
        for predecessor in lane.predecessor_ids:
            if predecessor not in lane_ids:
                errors.append(f"lane/{lane.lane_id}: unknown predecessor {predecessor}")

    return errors


def validate_trajectory(traj: TrajectoryState) -> list[str]:
    errors: list[str] = []
    t = len(traj.timestamps_s)
    n = len(traj.actor_ids)

    if traj.timestamps_s.shape != (t,):
        errors.append(f"timestamps_s must be [T], got {traj.timestamps_s.shape}")
    if traj.positions_xy.shape != (t, n, 2):
        errors.append(f"positions_xy must be [T,N,2], got {traj.positions_xy.shape}")
    if traj.headings_rad.shape != (t, n):
        errors.append(f"headings_rad must be [T,N], got {traj.headings_rad.shape}")
    if traj.valid.shape != (t, n):
        errors.append(f"valid must be [T,N], got {traj.valid.shape}")
    if len(set(traj.actor_ids)) != n:
        errors.append("actor_ids must be unique")
    if t > 1 and np.any(np.diff(traj.timestamps_s) <= 0):
        errors.append("timestamps_s must be strictly increasing")

    if traj.positions_xy.shape == (t, n, 2):
        bad = traj.valid & ~np.isfinite(traj.positions_xy).all(axis=-1)
        if np.any(bad):
            errors.append("valid trajectory entries contain non-finite positions")
    if traj.headings_rad.shape == (t, n):
        bad = traj.valid & ~np.isfinite(traj.headings_rad)
        if np.any(bad):
            errors.append("valid trajectory entries contain non-finite headings")

    return errors
