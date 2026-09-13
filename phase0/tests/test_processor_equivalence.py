import unittest
import numpy as np
from phase0_audit.native_common import CATEGORIES, config, replay
from phase0_audit.processor_equivalence import compare
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorElement as E, SledgeVectorRaw
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import sledge_raw_feature_processing
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_utils import coords_in_frame

class EquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config()

    def raw(self, category, states, mask_value=False):
        line = E(np.zeros((0, 0, 3), np.float32), np.zeros((0, 0), bool))
        actors = {cat:E(np.zeros((0, 5 if cat == 'static_objects' else 6), np.float32), np.zeros(0, bool)) for cat in CATEGORIES}
        actors[category] = E(states, np.full(len(states), mask_value, bool))
        return SledgeVectorRaw(lines=line, **actors, green_lights=line, red_lights=line,
                               ego=E(np.zeros(4), np.ones(1, bool)))

    def test_native_edge_cases(self):
        for cat in CATEGORIES:
            cap = getattr(self.cfg, 'num_'+cat)
            d = 5 if cat == 'static_objects' else 6
            for dtype in (np.float32, np.float64):
                for n in (0, cap, cap+1, cap+9):
                    for all_true in (False, True):
                        with self.subTest(category=cat, dtype=dtype, n=n, mask=all_true):
                            a = np.zeros((n, d), dtype)
                            a[:, 0] = 4
                            a[:, 2] = np.arange(n)*.01  # distinguish equal-distance actors
                            a[:, 3:5] = 1
                            if d == 6:
                                a[:, 5] = 99.123456789
                            raw = self.raw(cat, a, all_true)
                            out, _ = sledge_raw_feature_processing(raw, self.cfg)
                            self.assertIsNone(compare(raw, out, cat, self.cfg)[2])
                            self.assertEqual(int(out.__dict__[cat].mask.sum()), min(n, cap))

    def test_boundary_and_outside(self):
        for cat in CATEGORIES:
            d = 5 if cat == 'static_objects' else 6
            a = np.zeros((7, d), np.float32)
            x, y = np.asarray(self.cfg.frame)/2
            a[:, :2] = [[x,0],[-x,0],[0,y],[0,-y],[x,y],[x+.001,0],[0,-y-.001]]
            a[:, 3:5] = 1
            result = replay(a, cat, self.cfg)
            self.assertEqual(result['metrics']['in_frame_count'], 5)
            np.testing.assert_array_equal(result['inside'], coords_in_frame(a[:, :2], self.cfg.frame))
            raw = self.raw(cat, a)
            out, _ = sledge_raw_feature_processing(raw, self.cfg)
            self.assertIsNone(compare(raw, out, cat, self.cfg)[2])

    def test_comparison_detects_perturbed_state_and_mask(self):
        a = np.array([[0,0,0,1,2,3]], np.float32)
        raw = self.raw('vehicles', a)
        out, _ = sledge_raw_feature_processing(raw, self.cfg)
        out.vehicles.states[0,0] = .01
        self.assertIsNotNone(compare(raw, out, 'vehicles', self.cfg)[2])
        out.vehicles.mask[0] = False
        self.assertFalse(compare(raw, out, 'vehicles', self.cfg)[0]['mask_agreement'])

if __name__ == '__main__':
    unittest.main()
