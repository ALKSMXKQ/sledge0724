from __future__ import annotations

import pytest

from sledge.script.evaluation.phase2_b1_hazard_audit import summarize_audit


def _row(*, H: bool, status: str = "built", reason: str | None = None):
    return {
        "builder_status": status,
        "O": H,
        "E": H,
        "C": H,
        "T": H,
        "H": H,
        "failure_reason": reason or ("none" if H else "no_stable_reveal"),
    }


def test_failed_builder_stays_in_end_to_end_denominator():
    rows = [_row(H=True) for _ in range(18)] + [
        _row(H=False, status="builder_or_adapter_error", reason="builder_or_adapter_error")
        for _ in range(2)
    ]
    summary = summarize_audit(rows)

    assert summary["P_H_given_B1_built"] == pytest.approx(1.0)
    assert summary["P_H_end_to_end"] == pytest.approx(0.90)
    assert summary["builder_yield"] == pytest.approx(0.90)
    assert summary["gate"]["pass_target"] is True
    assert summary["gate"]["decision"] == "PROCEED_TO_DIFFUSION"


def test_below_90_percent_requires_builder_fix():
    rows = [_row(H=True) for _ in range(17)] + [_row(H=False) for _ in range(3)]
    summary = summarize_audit(rows)

    assert summary["P_H_end_to_end"] == pytest.approx(0.85)
    assert summary["gate"]["pass_target"] is False
    assert summary["gate"]["decision"] == "FIX_B1_BUILDER_BEFORE_DIFFUSION"


def test_95_percent_is_not_strictly_above_ideal_threshold():
    rows = [_row(H=True) for _ in range(19)] + [_row(H=False)]
    summary = summarize_audit(rows)

    assert summary["P_H_end_to_end"] == pytest.approx(0.95)
    assert summary["gate"]["pass_target"] is True
    assert summary["gate"]["pass_ideal"] is False
