import numpy as np

from sledge.hazard_equivalence.audit import structure_report


def test_structure_report_numpy():
    obj = {"actors": np.zeros((3, 11), dtype=np.float32), "name": "scene"}
    report = structure_report(obj)
    actors = report["items"]["actors"]
    assert actors["shape"] == [3, 11]
    assert actors["dtype"] == "float32"
