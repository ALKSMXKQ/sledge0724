from types import SimpleNamespace

import numpy as np
import pytest

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVectorElement,
    SledgeVectorRaw,
)
from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_clearance_ops import (
    HazardClearancePrimitiveOps,
)


def _scene() -> SledgeVectorRaw:
    lines = SledgeVectorElement(
        states=np.zeros((2, 20, 2), dtype=np.float32),
        mask=np.zeros((2, 20), dtype=bool),
    )
    # vehicle[0] overlaps reserved region; vehicle[1] is unrelated/far away.
    vehicles = SledgeVectorElement(
        states=np.asarray(
            [
                [10.0, 5.0, 0.0, 2.0, 4.8, 3.0],
                [35.0, -8.0, 0.0, 2.0, 4.8, 4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        mask=np.asarray([True, True, False]),
    )
    pedestrians = SledgeVectorElement(
        states=np.zeros((3, 6), dtype=np.float32),
        mask=np.zeros((3,), dtype=bool),
    )
    static_objects = SledgeVectorElement(
        states=np.zeros((3, 5), dtype=np.float32),
        mask=np.zeros((3,), dtype=bool),
    )
    lights = SledgeVectorElement(
        states=np.zeros((1, 5), dtype=np.float32),
        mask=np.zeros((1,), dtype=bool),
    )
    ego = SledgeVectorElement(
        states=np.asarray([8.0, 0.0, 0.0, 0.0], dtype=np.float32),
        mask=np.asarray([True]),
    )
    return SledgeVectorRaw(
        lines=lines,
        vehicles=vehicles,
        pedestrians=pedestrians,
        static_objects=static_objects,
        green_lights=lights,
        red_lights=SledgeVectorElement(
            states=np.zeros((1, 5), dtype=np.float32),
            mask=np.zeros((1,), dtype=bool),
        ),
        ego=ego,
    )


def _ctx():
    return SimpleNamespace(
        removed_vehicle_indices=[],
        extra={},
    )


def test_local_blocker_is_repositioned_before_deletion() -> None:
    scene = _scene()
    ops = HazardClearancePrimitiveOps()
    ctx = _ctx()
    reserved = (10.0, 5.0, 3.0, 2.5)
    before_x = float(scene.vehicles.states[0, 0])

    edits = ops._make_background_room(
        scene=scene,
        ctx=ctx,
        reserved_region=reserved,
        protected_refs=set(),
        side_sign=1.0,
        lane_y=0.0,
    )

    assert len(edits) == 1
    assert edits[0]["operation"] == "reposition"
    assert bool(scene.vehicles.mask[0]) is True
    assert float(scene.vehicles.states[0, 0]) != before_x
    assert ctx.extra["background_clearance_edit_count"] == 1
    assert ctx.extra["background_removal_count"] == 0


def test_blocker_is_deleted_when_relocation_has_no_solution() -> None:
    class NoRelocationOps(HazardClearancePrimitiveOps):
        def _find_background_relocation(self, **kwargs):
            return None

    scene = _scene()
    ops = NoRelocationOps()
    ctx = _ctx()
    reserved = (10.0, 5.0, 3.0, 2.5)

    edits = ops._make_background_room(
        scene=scene,
        ctx=ctx,
        reserved_region=reserved,
        protected_refs=set(),
        side_sign=1.0,
        lane_y=0.0,
    )

    assert len(edits) == 1
    assert edits[0]["operation"] == "delete"
    assert bool(scene.vehicles.mask[0]) is False
    assert ctx.removed_vehicle_indices == [0]
    assert ctx.extra["background_removal_count"] == 1


def test_protected_target_is_never_moved_or_deleted() -> None:
    scene = _scene()
    ops = HazardClearancePrimitiveOps()
    ctx = _ctx()
    reserved = (10.0, 5.0, 3.0, 2.5)

    edits = ops._make_background_room(
        scene=scene,
        ctx=ctx,
        reserved_region=reserved,
        protected_refs={("vehicles", 0)},
        side_sign=1.0,
        lane_y=0.0,
    )

    assert edits == []
    assert bool(scene.vehicles.mask[0]) is True
    assert float(scene.vehicles.states[0, 0]) == 10.0


def test_primitive_executor_uses_move_then_delete_ops() -> None:
    from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_executor import (
        PrimitiveExecutor,
    )

    executor = PrimitiveExecutor()
    assert isinstance(executor.ops, HazardClearancePrimitiveOps)


def test_timing_aware_layout_synchronizes_lane_entry_and_ego_arrival() -> None:
    from sledge.semantic_control.occluded_pedestrian_pipeline.generation.hazard_spec import (
        HazardSemanticSpec,
    )
    from sledge.semantic_control.occluded_pedestrian_pipeline.generation.primitive_ops import (
        OCCLUDER_SPECS,
    )

    scene = _scene()
    # Remove the two background vehicles from the placement test and activate a
    # controlled pedestrian slot.
    scene.vehicles.mask[:] = False
    scene.pedestrians.mask[0] = True
    scene.pedestrians.states[0] = np.asarray(
        [12.0, -5.0, np.pi / 2.0, 0.75, 0.75, 1.6],
        dtype=np.float32,
    )

    spec = HazardSemanticSpec(spec_id="timing-aware")
    spec.road_layer.lane_width_m = 3.5
    spec.interaction_layer.conflict_type = "lateral_conflict"
    spec.interaction_layer.conflict_direction = "right_to_left"
    spec.risk_layer.risk_level = "moderate"
    spec.risk_layer.ttc_range_s = (2.0, 3.0)
    spec.risk_layer.longitudinal_distance_range_m = (10.0, 18.0)
    spec.risk_layer.target_actor_speed_mps = 1.6

    ctx = SimpleNamespace(
        spec=spec,
        actor_elem_name="pedestrians",
        actor_index=0,
        anchor={"ego_speed": 7.0, "lane_y": 0.8},
        extra={},
        removed_vehicle_indices=[],
        notes=[],
    )

    ops = HazardClearancePrimitiveOps()
    layout = ops._plan_occluded_actor_layout(
        scene=scene,
        ctx=ctx,
        occluder_spec=OCCLUDER_SPECS["vehicle"],
        occluder_index=0,
        params={
            "direction": "right_to_left",
            "target_actor_speed_mps": 1.6,
            "frame0_time_offset_s": 2.1,
            "compensate_frame0_offset": True,
        },
    )

    assert layout["lane_center_y"] == 0.0
    assert 2.0 <= layout["target_interaction_ttc_s"] <= 3.0
    assert layout["pedestrian_lane_entry_time_s"] == pytest.approx(
        layout["target_interaction_ttc_s"], abs=1e-5
    )
    assert layout["ego_arrival_time_s"] == pytest.approx(
        layout["pedestrian_lane_entry_time_s"], abs=1e-5
    )
    assert layout["arrival_time_error_s"] == pytest.approx(0.0, abs=1e-6)
    assert layout["occluder_lane_boundary_gap_m"] >= 1.50
    assert ctx.anchor["lane_y"] == pytest.approx(0.0)