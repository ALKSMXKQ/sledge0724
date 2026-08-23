"""Fixed TensorRT deployment contract for the SLEDGE RVAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


INPUT_NAME = "raster"
INPUT_SHAPE = (1, 12, 256, 256)
OUTPUT_NAMES = (
    "lines_states",
    "lines_logits",
    "vehicles_states",
    "vehicles_logits",
    "pedestrians_states",
    "pedestrians_logits",
    "static_objects_states",
    "static_objects_logits",
    "green_lights_states",
    "green_lights_logits",
    "red_lights_states",
    "red_lights_logits",
    "ego_states",
    "ego_mask",
)

OUTPUT_SHAPES: Dict[str, Tuple[int, ...]] = {
    "lines_states": (1, 50, 20, 2),
    "lines_logits": (1, 50),
    "vehicles_states": (1, 50, 6),
    "vehicles_logits": (1, 50),
    "pedestrians_states": (1, 20, 6),
    "pedestrians_logits": (1, 20),
    "static_objects_states": (1, 30, 5),
    "static_objects_logits": (1, 30),
    "green_lights_states": (1, 20, 20, 2),
    "green_lights_logits": (1, 20),
    "red_lights_states": (1, 20, 20, 2),
    "red_lights_logits": (1, 20),
    "ego_states": (1, 1),
    "ego_mask": (1, 1),
}

MASK_PREFIXES = (
    "lines",
    "vehicles",
    "pedestrians",
    "static_objects",
    "green_lights",
    "red_lights",
)


@dataclass(frozen=True)
class Tolerances:
    max_abs: float
    mean_abs: float


TOLERANCES = {
    # CPU kernels in PyTorch and ONNX Runtime use different reduction orders in
    # the six-layer transformer. This bound is still ~1e-5 relative to the
    # 32-metre coordinate range and preserves identical postprocessed queries.
    "fp32": Tolerances(max_abs=5.0e-4, mean_abs=5.0e-5),
    "fp16": Tolerances(max_abs=5.0e-2, mean_abs=5.0e-3),
}


def validate_shapes(outputs: Dict[str, object]) -> None:
    """Raise a useful error if an output violates the fixed contract."""
    missing = set(OUTPUT_NAMES) - set(outputs)
    extra = set(outputs) - set(OUTPUT_NAMES)
    if missing or extra:
        raise ValueError(f"Output names mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for name, expected in OUTPUT_SHAPES.items():
        actual = tuple(outputs[name].shape)  # type: ignore[attr-defined]
        if actual != expected:
            raise ValueError(f"{name}: expected shape {expected}, got {actual}")
