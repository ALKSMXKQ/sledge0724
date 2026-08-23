"""Raw and postprocessed cross-backend consistency metrics."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from .contract import MASK_PREFIXES, OUTPUT_NAMES, Tolerances, validate_shapes


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.where(value >= 0, 1.0 / (1.0 + np.exp(-value)), np.exp(value) / (1.0 + np.exp(value)))


def compare_outputs(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    threshold: float,
    tolerances: Tolerances,
) -> Dict[str, object]:
    validate_shapes(dict(reference))
    validate_shapes(dict(candidate))
    per_output: Dict[str, object] = {}
    global_max = 0.0
    weighted_abs_sum = 0.0
    value_count = 0
    for name in OUTPUT_NAMES:
        ref = np.asarray(reference[name], dtype=np.float64)
        got = np.asarray(candidate[name], dtype=np.float64)
        diff = np.abs(ref - got)
        max_abs = float(diff.max(initial=0.0))
        mean_abs = float(diff.mean())
        per_output[name] = {"max_abs": max_abs, "mean_abs": mean_abs}
        global_max = max(global_max, max_abs)
        weighted_abs_sum += float(diff.sum())
        value_count += diff.size

    postprocessed: Dict[str, object] = {}
    for prefix in MASK_PREFIXES:
        ref_prob = sigmoid(reference[f"{prefix}_logits"])
        got_prob = sigmoid(candidate[f"{prefix}_logits"])
        ref_active = ref_prob >= threshold
        got_active = got_prob >= threshold
        union = np.logical_or(ref_active, got_active)
        intersection = np.logical_and(ref_active, got_active)
        state_diff = np.abs(
            np.asarray(reference[f"{prefix}_states"], dtype=np.float64)
            - np.asarray(candidate[f"{prefix}_states"], dtype=np.float64)
        )
        query_mask = np.squeeze(union, axis=0)
        selected = state_diff[0][query_mask]
        postprocessed[prefix] = {
            "reference_count": int(ref_active.sum()),
            "candidate_count": int(got_active.sum()),
            "active_query_iou": float(intersection.sum() / union.sum()) if union.any() else 1.0,
            "max_abs_state_on_union": float(selected.max(initial=0.0)),
            "mean_abs_probability": float(np.abs(ref_prob - got_prob).mean()),
        }

    global_mean = weighted_abs_sum / max(value_count, 1)
    passed = global_max <= tolerances.max_abs and global_mean <= tolerances.mean_abs
    return {
        "passed": passed,
        "thresholds": {"max_abs": tolerances.max_abs, "mean_abs": tolerances.mean_abs},
        "global": {"max_abs": global_max, "mean_abs": global_mean},
        "raw_outputs": per_output,
        "postprocessed": postprocessed,
    }

