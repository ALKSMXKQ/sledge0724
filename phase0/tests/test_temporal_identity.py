import unittest
from types import SimpleNamespace as NS
import numpy as np
from phase0_audit.temporal_state import local_state, transitions, track_summary, cache_match

class TemporalTests(unittest.TestCase):
    def test_transform_uses_center_and_heading_wrap(self):
        ego=NS(center=NS(x=10.,y=20.,heading=np.pi/2))
        actor=NS(center=NS(x=10.,y=22.,heading=-np.pi),box=NS(width=1.,length=2.),velocity=NS(x=3.,y=4.))
        np.testing.assert_allclose(local_state(actor,ego),[2,0,np.pi/2,1,2,5],atol=1e-12)

    def test_exit_causes_are_distinct(self):
        previous={'a':dict(inside_frame=True,status='kept'), 'b':dict(inside_frame=True,status='kept'), 'c':dict(inside_frame=True,status='kept')}
        current={'a':dict(inside_frame=False,status='outside_frame'),'b':dict(inside_frame=True,status='cap_excluded')}
        self.assertEqual({r['track_token']:r['event'] for r in transitions(previous,current)},
            {'a':'spatial_exit','b':'capacity_exclusion','c':'observation_disappearance_unknown_cause'})

    def test_detection_tokens_can_change_without_track_discontinuity(self):
        rows=[dict(frame_index=i,track_token='stable',actor_type='VEHICLE',detection_token=str(i),runtime_track_id=9,
                   timestamp_us=i*50000,representation_status='kept') for i in [0,1,2,4,5]]
        r=track_summary(rows)
        self.assertEqual(r['unique_runtime_track_ids'],1)
        self.assertEqual(r['observation_gap_count'],1)
        self.assertEqual(r['max_consecutive_frames'],3)
        self.assertEqual(r['detection_token_changes'],4)

    def test_ambiguous_cache_matching_is_explicit(self):
        a=NS(states=np.zeros((2,6),np.float32))
        r=cache_match(a,a,['a','b'])
        self.assertEqual(r['ambiguous_count'],2)
        b=NS(states=np.ones((2,6),np.float32))
        self.assertFalse(cache_match(a,b,['a','b'])['passed'])

class VelocityDiagnosticsTests(unittest.TestCase):
    def test_static_dummy_velocity_is_excluded(self):
        from phase0_audit.temporal_diagnostics import velocity_pair
        a=dict(frame_index=0,timestamp_us=0,global_x=0,global_y=0,global_heading=0,vx=0,vy=0,actor_type='BARRIER')
        b=dict(a,frame_index=1,timestamp_us=50000,global_x=1)
        self.assertIsNone(velocity_pair(a,b,{'VEHICLE','PEDESTRIAN'}))
        a['actor_type']='VEHICLE'
        self.assertEqual(velocity_pair(a,b,{'VEHICLE'})['global_velocity_residual_mps'],20.)

if __name__ == '__main__':
    unittest.main()
