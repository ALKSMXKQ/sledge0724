"""Fail-closed matching for controlled B1 objects after SLEDGE preprocessing.

SLEDGE raw feature processing sorts agents by distance, clips each collection to
its fixed capacity, and may therefore change slot indices.  The historical
matching helpers always returned the nearest-looking processed slot whenever at
least one candidate existed.  If the controlled pedestrian or occluder was
actually clipped out, that behavior could silently relabel an unrelated
background object as the controlled hazard entity.

This helper deliberately fails closed.  It matches only geometry fields that
SLEDGE preprocessing copies without semantic transformation (x, y, heading,
width, length), excludes velocity because preprocessing may clamp it to the
model input limit, and returns -1 unless the best candidate is effectively the
same object.
"""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_MAX_WEIGHTED_ERROR = 1e-3


def strict_match_processed_slot(
    raw_elem: Any,
    raw_index: int,
    vector_elem: Any,
    *,
    max_weighted_error: float = DEFAULT_MAX_WEIGHTED_ERROR,
) -> int:
    """Return the processed slot for one raw entity, or ``-1`` if not retained.

    The matcher intentionally ignores velocity.  ``process_agents`` may clamp
    velocity to the configured model maximum, while x/y/heading/width/length
    are copied into the processed vector for retained entities.
    """

    raw_states = np.asarray(raw_elem.states)
    if raw_states.ndim == 1:
        raw_states = raw_states.reshape(1, -1)
    if raw_index < 0 or raw_index >= len(raw_states):
        return -1

    target = np.asarray(raw_states[raw_index], dtype=np.float32).reshape(-1)
    states = np.asarray(vector_elem.states)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    masks = np.asarray(vector_elem.mask).reshape(-1)
    usable = min(len(states), len(masks))
    if usable <= 0:
        return -1

    states = states[:usable]
    masks = masks[:usable]
    active = (
        masks.astype(bool)
        if masks.dtype == np.bool_
        else masks.astype(np.float32) >= 0.3
    )
    valid = np.where(active)[0]
    if not len(valid):
        return -1

    width = min(5, states.shape[-1], target.shape[-1])
    if width <= 0:
        return -1

    # x/y dominate; heading and box dimensions are weaker tie-breakers.
    scales = np.asarray(
        [1.0, 1.0, 0.5, 0.25, 0.25],
        dtype=np.float32,
    )[:width]
    errors = np.linalg.norm(
        (states[valid, :width].astype(np.float32) - target[:width]) * scales,
        axis=1,
    )
    best_position = int(np.argmin(errors))
    best_error = float(errors[best_position])
    if not np.isfinite(best_error) or best_error > float(max_weighted_error):
        return -1
    return int(valid[best_position])


__all__ = [
    "DEFAULT_MAX_WEIGHTED_ERROR",
    "strict_match_processed_slot",
]
