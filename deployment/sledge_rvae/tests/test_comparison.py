import unittest

import numpy as np

from deployment.sledge_rvae.python.comparison import compare_outputs
from deployment.sledge_rvae.python.contract import OUTPUT_SHAPES, TOLERANCES


class ComparisonTest(unittest.TestCase):
    def make_outputs(self):
        return {name: np.zeros(shape, dtype=np.float32) for name, shape in OUTPUT_SHAPES.items()}

    def test_identical_outputs_pass(self):
        outputs = self.make_outputs()
        report = compare_outputs(outputs, outputs, 0.3, TOLERANCES["fp32"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["global"]["max_abs"], 0.0)
        self.assertEqual(report["postprocessed"]["vehicles"]["active_query_iou"], 1.0)

    def test_error_fails(self):
        reference = self.make_outputs()
        candidate = self.make_outputs()
        candidate["vehicles_states"][0, 0, 0] = 1.0
        report = compare_outputs(reference, candidate, 0.3, TOLERANCES["fp32"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["global"]["max_abs"], 1.0)


if __name__ == "__main__":
    unittest.main()

