import unittest
from phase0_audit.scenario_index import classify_unknown
from phase0_audit.native_common import balanced_sample
from phase0_audit.targeted_retention import aggregate

class TargetedTests(unittest.TestCase):
    def test_unknown_is_not_invented(self):
        self.assertEqual(classify_unknown([], True), ('unknown', 'unlabeled_in_authoritative_db'))
        self.assertEqual(classify_unknown(['a','b'], True)[0], 'unknown')
        self.assertEqual(classify_unknown(['a'], True)[0], 'a')
        self.assertEqual(classify_unknown(['a'], False)[0], 'unknown')

    def test_sampling_is_order_independent(self):
        rows = [dict(path=str(i), log_name=str(i%3)) for i in range(30)]
        a = balanced_sample(rows, 9, 7)
        self.assertEqual(a, balanced_sample(list(reversed(rows)), 9, 7))
        self.assertEqual(len(set(r['path'] for r in a)),9)
        self.assertEqual([sum(r['log_name']==str(i) for r in a) for i in range(3)], [3,3,3])

    def test_capacity_denominator_and_null_distances(self):
        row = dict(raw_count=100,in_frame_count=10,kept_count=10,dropped_by_cap=0,
                   dropped_outside_frame=90,overflow=-10,d_K=3,d_first_drop=None,boundary_gap=None)
        result = aggregate([row])
        self.assertEqual(result['p_kept_given_inside'],1)
        self.assertEqual(result['d_first_drop'],{'n':0})
        self.assertEqual(result['positive_overflow'],{'n':0})
        self.assertEqual(result['overflow']['median'],-10)

if __name__ == '__main__':
    unittest.main()
