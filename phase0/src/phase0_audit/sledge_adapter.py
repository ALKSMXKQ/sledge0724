from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .canonical import ActorState, ActorType, LaneState, SceneState
from .sledge_contract import AGENT_INDEX, STATIC_OBJECT_INDEX, EGO_INDEX


class SledgeProcessedVectorAdapter:
    """Adapter for the real processed ``SledgeVector`` representation.

    Important audited limitations:
    - actor track IDs are no longer present, so canonical IDs are slot-based;
    - processed line vectors preserve geometry only, not original lane IDs/topology.

    The implementation deliberately uses duck typing so Phase 0 remains standalone and
    does not import the SLEDGE package during ordinary unit tests.
    """

    frame = "ego_local"

    def to_canonical(self, vector: Any, *, scene_id: str) -> SceneState:
        actors: list[ActorState] = []
        self._append_agents(actors, vector.vehicles, ActorType.VEHICLE, "vehicle")
        self._append_agents(actors, vector.pedestrians, ActorType.PEDESTRIAN, "pedestrian")
        self._append_static(actors, vector.static_objects)

        lanes = self._geometry_only_lines(vector.lines)
        metadata = {"representation": "SledgeVector", "identity_preserved": False, "lane_topology_preserved": False}

        ego_states = np.asarray(vector.ego.states)
        if ego_states.size >= 4:
            flat = ego_states.reshape(-1)
            metadata["ego_velocity_xy"] = [float(flat[EGO_INDEX["velocity_x"]]), float(flat[EGO_INDEX["velocity_y"]])]
            metadata["ego_acceleration_xy"] = [float(flat[EGO_INDEX["acceleration_x"]]), float(flat[EGO_INDEX["acceleration_y"]])]

        return SceneState(scene_id=str(scene_id), actors=actors, lanes=lanes, frame=self.frame, metadata=metadata)

    def from_canonical(self, scene: SceneState, *, template: Any) -> Any:
        """Write canonical actor fields back into a deep-copied SledgeVector template.

        Only fields represented by SLEDGE are written. Unknown/padded slots and all line,
        traffic-light and ego data are preserved from ``template``.
        """
        restored = copy.deepcopy(template)
        category_to_element = {
            "vehicle": restored.vehicles,
            "pedestrian": restored.pedestrians,
            "static": restored.static_objects,
        }

        for actor in scene.actors:
            category = actor.metadata.get("sledge_category")
            slot = actor.metadata.get("sledge_slot")
            if category not in category_to_element or slot is None:
                raise ValueError("Canonical actor is missing SLEDGE category/slot metadata")
            slot = int(slot)
            element = category_to_element[category]
            states = element.states

            if category in {"vehicle", "pedestrian"}:
                states[slot, AGENT_INDEX["x"]] = actor.x
                states[slot, AGENT_INDEX["y"]] = actor.y
                states[slot, AGENT_INDEX["heading"]] = actor.heading_rad
                states[slot, AGENT_INDEX["width"]] = actor.width_m
                states[slot, AGENT_INDEX["length"]] = actor.length_m
                states[slot, AGENT_INDEX["speed"]] = actor.speed_mps
            else:
                states[slot, STATIC_OBJECT_INDEX["x"]] = actor.x
                states[slot, STATIC_OBJECT_INDEX["y"]] = actor.y
                states[slot, STATIC_OBJECT_INDEX["heading"]] = actor.heading_rad
                states[slot, STATIC_OBJECT_INDEX["width"]] = actor.width_m
                states[slot, STATIC_OBJECT_INDEX["length"]] = actor.length_m

        return restored

    @staticmethod
    def _valid_indices(element: Any) -> np.ndarray:
        states = np.asarray(element.states)
        mask = np.asarray(element.mask).astype(bool).reshape(-1)
        if states.ndim != 2:
            raise ValueError("processed actor/static states must be rank-2, got %r" % (states.shape,))
        if len(mask) != len(states):
            raise ValueError("mask/state slot count mismatch: %d vs %d" % (len(mask), len(states)))
        return np.flatnonzero(mask)

    def _append_agents(self, out: list[ActorState], element: Any, actor_type: ActorType, category: str) -> None:
        states = np.asarray(element.states)
        for slot in self._valid_indices(element):
            row = states[slot]
            heading = float(row[AGENT_INDEX["heading"]])
            speed = float(row[AGENT_INDEX["speed"]])
            velocity = np.array([speed * np.cos(heading), speed * np.sin(heading)], dtype=np.float64)
            out.append(ActorState(
                track_id="%s:%d" % (category, slot),
                actor_type=actor_type,
                position_xy=[row[AGENT_INDEX["x"]], row[AGENT_INDEX["y"]]],
                heading_rad=heading,
                velocity_xy=velocity,
                width_m=row[AGENT_INDEX["width"]],
                length_m=row[AGENT_INDEX["length"]],
                valid=True,
                source_index=int(slot),
                metadata={"sledge_category": category, "sledge_slot": int(slot), "identity_source": "slot_only"},
            ))

    def _append_static(self, out: list[ActorState], element: Any) -> None:
        states = np.asarray(element.states)
        for slot in self._valid_indices(element):
            row = states[slot]
            out.append(ActorState(
                track_id="static:%d" % slot,
                actor_type=ActorType.STATIC,
                position_xy=[row[STATIC_OBJECT_INDEX["x"]], row[STATIC_OBJECT_INDEX["y"]]],
                heading_rad=row[STATIC_OBJECT_INDEX["heading"]],
                velocity_xy=[0.0, 0.0],
                width_m=row[STATIC_OBJECT_INDEX["width"]],
                length_m=row[STATIC_OBJECT_INDEX["length"]],
                valid=True,
                source_index=int(slot),
                metadata={"sledge_category": "static", "sledge_slot": int(slot), "identity_source": "slot_only"},
            ))

    @staticmethod
    def _geometry_only_lines(element: Any) -> list[LaneState]:
        states = np.asarray(element.states)
        mask = np.asarray(element.mask).astype(bool).reshape(-1)
        if states.ndim != 3 or states.shape[-1] != 2:
            raise ValueError("processed line states must be [N,P,2], got %r" % (states.shape,))
        lanes: list[LaneState] = []
        for slot in np.flatnonzero(mask):
            centerline = states[slot]
            if len(centerline) < 2:
                continue
            lanes.append(LaneState(
                lane_id="geometry_line:%d" % slot,
                centerline_xy=centerline,
                successor_ids=(),
                predecessor_ids=(),
                metadata={"sledge_slot": int(slot), "topology_preserved": False, "semantic_kind": "summarized_line_geometry"},
            ))
        return lanes
