from sledge.hazard_equivalence.ground_truth import TimingClass, compute_critical_timing


def test_critical_timing_classes():
    assert compute_critical_timing(1.0, 1.8).timing_class == TimingClass.AGGRESSIVE
    assert compute_critical_timing(1.0, 3.0).timing_class == TimingClass.MODERATE
    assert compute_critical_timing(1.0, 6.0).timing_class == TimingClass.SAFE
