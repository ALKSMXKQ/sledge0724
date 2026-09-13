from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


# Verified from the repository source. These are not guessed indices.
AGENT_INDEX: Dict[str, int] = {
    "x": 0,
    "y": 1,
    "heading": 2,
    "width": 3,
    "length": 4,
    "speed": 5,
}

STATIC_OBJECT_INDEX: Dict[str, int] = {
    "x": 0,
    "y": 1,
    "heading": 2,
    "width": 3,
    "length": 4,
}

EGO_INDEX: Dict[str, int] = {
    "velocity_x": 0,
    "velocity_y": 1,
    "acceleration_x": 2,
    "acceleration_y": 3,
}

RASTER_CHANNELS: Dict[str, int] = {
    "line_x": 0,
    "line_y": 1,
    "vehicle_x": 2,
    "vehicle_y": 3,
    "pedestrian_x": 4,
    "pedestrian_y": 5,
    "static_object_x": 6,
    "static_object_y": 7,
    "green_light_x": 8,
    "green_light_y": 9,
    "red_light_x": 10,
    "red_light_y": 11,
}

DEFAULT_RVAE_CONTRACT: Dict[str, Any] = {
    "frame_m": [64, 64],
    "pixel_size_m": 0.25,
    "pixel_frame": [256, 256],
    "num_input_channels": 12,
    "down_factor": 32,
    "latent_channels": 64,
    "latent_frame": [8, 8],
    "latent_source_for_diffusion": "mu",
    "latent_scaling_before_diffusion": "none_observed_in_dataset_or_training_loader",
}

DEFAULT_DIT_CONTRACT: Dict[str, Any] = {
    "in_channels": 64,
    "out_channels": 64,
    "sample_size": 8,
    "patch_size": 1,
    "num_layers": 12,
    "num_attention_heads": 12,
}

DEFAULT_DDPM_CONTRACT: Dict[str, Any] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.0015,
    "beta_end": 0.015,
    "beta_schedule": "linear",
    "clip_sample": False,
}


@dataclass
class RepositoryContractReport:
    ok: bool
    checks: Dict[str, bool]
    errors: List[str]
    observed: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def verified_contract() -> Dict[str, Any]:
    """Return the source-audited representation contract as plain data."""
    return {
        "agent_index": dict(AGENT_INDEX),
        "static_object_index": dict(STATIC_OBJECT_INDEX),
        "ego_index": dict(EGO_INDEX),
        "raster_channels": dict(RASTER_CHANNELS),
        "rvae": dict(DEFAULT_RVAE_CONTRACT),
        "dit": dict(DEFAULT_DIT_CONTRACT),
        "ddpm": dict(DEFAULT_DDPM_CONTRACT),
        "semantic_notes": {
            "agent_coordinate_frame": "ego_local",
            "agent_heading_unit": "radian",
            "agent_speed_representation": "scalar_magnitude_mps",
            "agent_size_representation": "full_width_and_full_length_m",
            "processed_agent_mask": "True_means_valid_slot",
            "raw_agent_mask": "allocated_as_all_False_and_not_used_by_process_agents",
            "raw_line_state": "x_y_heading",
            "processed_line_state": "x_y_only",
            "processed_line_topology": "lane_ids_and_successor_ids_not_preserved",
            "processed_actor_identity": "original_nuplan_track_id_not_preserved",
        },
    }


def verify_repository_contract() -> RepositoryContractReport:
    """Verify the audited constants against the currently checked-out SLEDGE code.

    This is deliberately a runtime assertion against the real repository classes.  It
    protects Phase 0 from silently becoming stale if the upstream representation changes.
    """

    checks: Dict[str, bool] = {}
    errors: List[str] = []
    observed: Dict[str, Any] = {}

    try:
        from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
            AgentIndex,
            EgoIndex,
            StaticObjectIndex,
        )
        from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRasterIndex
        from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
    except Exception as exc:  # pragma: no cover - depends on full SLEDGE environment
        return RepositoryContractReport(
            ok=False,
            checks={},
            errors=["Could not import SLEDGE representation classes: %r" % (exc,)],
            observed={},
        )

    def check(name: str, actual: Any, expected: Any) -> None:
        observed[name] = _jsonable(actual)
        passed = _jsonable(actual) == _jsonable(expected)
        checks[name] = passed
        if not passed:
            errors.append("%s: expected %r, observed %r" % (name, expected, actual))

    check("AgentIndex.X", AgentIndex.X, AGENT_INDEX["x"])
    check("AgentIndex.Y", AgentIndex.Y, AGENT_INDEX["y"])
    check("AgentIndex.HEADING", AgentIndex.HEADING, AGENT_INDEX["heading"])
    check("AgentIndex.WIDTH", AgentIndex.WIDTH, AGENT_INDEX["width"])
    check("AgentIndex.LENGTH", AgentIndex.LENGTH, AGENT_INDEX["length"])
    check("AgentIndex.VELOCITY", AgentIndex.VELOCITY, AGENT_INDEX["speed"])
    check("AgentIndex.size", AgentIndex.size(), 6)

    check("StaticObjectIndex.X", StaticObjectIndex.X, STATIC_OBJECT_INDEX["x"])
    check("StaticObjectIndex.Y", StaticObjectIndex.Y, STATIC_OBJECT_INDEX["y"])
    check("StaticObjectIndex.HEADING", StaticObjectIndex.HEADING, STATIC_OBJECT_INDEX["heading"])
    check("StaticObjectIndex.WIDTH", StaticObjectIndex.WIDTH, STATIC_OBJECT_INDEX["width"])
    check("StaticObjectIndex.LENGTH", StaticObjectIndex.LENGTH, STATIC_OBJECT_INDEX["length"])
    check("StaticObjectIndex.size", StaticObjectIndex.size(), 5)

    check("EgoIndex.VELOCITY_X", EgoIndex.VELOCITY_X, EGO_INDEX["velocity_x"])
    check("EgoIndex.VELOCITY_Y", EgoIndex.VELOCITY_Y, EGO_INDEX["velocity_y"])
    check("EgoIndex.ACCELERATION_X", EgoIndex.ACCELERATION_X, EGO_INDEX["acceleration_x"])
    check("EgoIndex.ACCELERATION_Y", EgoIndex.ACCELERATION_Y, EGO_INDEX["acceleration_y"])
    check("EgoIndex.size", EgoIndex.size(), 4)

    check("SledgeRasterIndex.size", SledgeRasterIndex.size(), 12)
    check("SledgeRasterIndex.LINE_X", SledgeRasterIndex.LINE_X, RASTER_CHANNELS["line_x"])
    check("SledgeRasterIndex.LINE_Y", SledgeRasterIndex.LINE_Y, RASTER_CHANNELS["line_y"])
    check("SledgeRasterIndex.VEHICLE_X", SledgeRasterIndex.VEHICLE_X, RASTER_CHANNELS["vehicle_x"])
    check("SledgeRasterIndex.VEHICLE_Y", SledgeRasterIndex.VEHICLE_Y, RASTER_CHANNELS["vehicle_y"])
    check("SledgeRasterIndex.PEDESTRIAN_X", SledgeRasterIndex.PEDESTRIAN_X, RASTER_CHANNELS["pedestrian_x"])
    check("SledgeRasterIndex.PEDESTRIAN_Y", SledgeRasterIndex.PEDESTRIAN_Y, RASTER_CHANNELS["pedestrian_y"])

    config = RVAEConfig()
    check("RVAE.frame", list(config.frame), DEFAULT_RVAE_CONTRACT["frame_m"])
    check("RVAE.pixel_size", config.pixel_size, DEFAULT_RVAE_CONTRACT["pixel_size_m"])
    check("RVAE.pixel_frame", list(config.pixel_frame), DEFAULT_RVAE_CONTRACT["pixel_frame"])
    check("RVAE.num_input_channels", config.num_input_channels, DEFAULT_RVAE_CONTRACT["num_input_channels"])
    check("RVAE.latent_channel", config.latent_channel, DEFAULT_RVAE_CONTRACT["latent_channels"])
    check("RVAE.latent_frame", list(config.latent_frame), DEFAULT_RVAE_CONTRACT["latent_frame"])

    return RepositoryContractReport(
        ok=all(checks.values()) and not errors,
        checks=checks,
        errors=errors,
        observed=observed,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
