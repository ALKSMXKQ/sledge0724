from sledge.script.evaluation.phase1_hazard_ground_truth import RevealConfig, detect_reveal


def test_reveal_requires_k_consecutive_visible_frames():
    cfg = RevealConfig(visibility_threshold=0.5, consecutive_visible_frames=3)
    r = detect_reveal([0.0, 0.1, 0.8, 0.9, 0.95], [0, 1, 2, 3, 4], cfg)
    assert r.reveal_exists and r.reveal_time_s == 2.0


def test_visibility_flicker_is_not_reveal():
    cfg = RevealConfig(visibility_threshold=0.5, consecutive_visible_frames=3)
    r = detect_reveal([0.0, 0.8, 0.2, 0.9, 0.2], [0, 1, 2, 3, 4], cfg)
    assert not r.reveal_exists


def test_threshold_is_strict():
    cfg = RevealConfig(visibility_threshold=0.5, consecutive_visible_frames=2)
    r = detect_reveal([0.0, 0.5, 0.8, 0.9], [0, 1, 2, 3], cfg)
    assert not r.reveal_exists
