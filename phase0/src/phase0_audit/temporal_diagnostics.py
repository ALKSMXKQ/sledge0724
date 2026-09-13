"""Summarize saved temporal evidence, separating measured agent and dummy static velocity."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import inspect
import json
from pathlib import Path
import numpy as np
from .native_common import REPO, dump, quantiles, sha


def velocity_pair(previous, current, dynamic_names):
    if current['frame_index'] != previous['frame_index']+1 or previous['actor_type'] not in dynamic_names:
        return None
    dt=(current['timestamp_us']-previous['timestamp_us'])/1e6
    displacement=np.array([current['global_x']-previous['global_x'], current['global_y']-previous['global_y']])/dt
    velocity=np.array([previous['vx'],previous['vy']])
    c,s=np.cos(previous['global_heading']),np.sin(previous['global_heading'])
    rotated=np.array([c*velocity[0]-s*velocity[1],s*velocity[0]+c*velocity[1]])
    return dict(global_velocity_residual_mps=float(np.linalg.norm(displacement-velocity)),
                body_velocity_hypothesis_residual_mps=float(np.linalg.norm(displacement-rotated)),
                displacement_mps=displacement.tolist(), recorded_velocity_mps=velocity.tolist(), dt_s=dt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=REPO/'phase0/outputs/temporal_identity')
    p.add_argument('--output-dir',type=Path,default=REPO/'phase0/outputs/temporal_identity')
    args=p.parse_args()
    path=args.output_dir/'diagnostic_summary.json'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
    from nuplan.common.actor_state.static_object import StaticObject
    names={t.name for t in AGENT_TYPES}
    previous={}; by_type=defaultdict(list); examples=[]; observations=Counter()
    with gzip.open(args.input_dir/'actor_observations.csv.gz','rt') as f:
        for row in csv.DictReader(f):
            for key in ('frame_index','timestamp_us'):
                row[key]=int(row[key])
            for key in ('global_x','global_y','global_heading','vx','vy'):
                row[key]=float(row[key]) if row[key] else None
            key=(row['log_name'],row['token'],row['track_token'])
            observations['dynamic_measured' if row['actor_type'] in names else 'static_dummy_zero']+=1
            if key in previous:
                result=velocity_pair(previous[key],row,names)
                if result is not None:
                    by_type[row['actor_type']].append(result)
                    examples.append(dict(scenario_token=row['token'],log_name=row['log_name'],track_token=row['track_token'],
                        previous_detection_token=previous[key]['detection_token'],detection_token=row['detection_token'],
                        timestamp_us=row['timestamp_us'],actor_type=row['actor_type'],**result))
            previous[key]=row
    counts=Counter()
    with (args.input_dir/'transitions.csv').open() as f:
        for row in csv.DictReader(f):
            if row['previous_status']=='kept':
                counts[row['current_status'] or 'observation_disappearance_unknown_cause']+=1
    scenarios=json.loads((args.input_dir/'per_scenario.json').read_text())
    weighted={}
    for field in ('ego_velocity_global_frame_residual_mps','ego_velocity_rotated_from_local_residual_mps','ego_pose_minus_lidar_timestamp_us'):
        vals=[s[field] for s in scenarios];n=sum(v['n'] for v in vals)
        mean=sum(v['mean']*v['n'] for v in vals)/n
        weighted[field]=dict(n=n,mean=mean,min=min(v['min'] for v in vals),max=max(v['max'] for v in vals))
    result=dict(source_observations_sha256=sha(args.input_dir/'actor_observations.csv.gz'),
        static_velocity_source=inspect.getfile(StaticObject),static_velocity_source_sha256=sha(inspect.getfile(StaticObject)),
        static_velocity_note='StaticObject exposes synthetic (0,0), not a measured velocity. Original all-object residual includes it; use the dynamic-only distributions here for measured agent velocity.',
        observations=dict(observations),by_dynamic_actor_type={typ:{key:quantiles([r[key] for r in rows]) for key in
            ('global_velocity_residual_mps','body_velocity_hypothesis_residual_mps')} for typ,rows in by_type.items()},
        largest_dynamic_velocity_residuals=sorted(examples,key=lambda r:r['global_velocity_residual_mps'],reverse=True)[:10],
        kept_representation_departures=dict(counts),**weighted,
        cached_raw_actor_matches=sum(s['cache_matches'][c]['cache_count'] for s in scenarios for c in s['cache_matches']),
        cached_raw_max_abs_error=max(s['cache_matches'][c].get('max_abs_error',0) for s in scenarios for c in s['cache_matches']),
        cache_ambiguities=sum(s['cache_matches'][c]['ambiguous_count'] for s in scenarios for c in s['cache_matches']),
        interpretation='Ego velocity favors body axes; agent velocity favors global axes. Forward differences are diagnostic and need not equal smoothed annotated velocity. Large residuals are data-quality observations, not proven ID switches or model failures.')
    dump(path,result)
    print({k:v for k,v in result.items() if k not in ('largest_dynamic_velocity_residuals','source_observations_sha256')})

if __name__=='__main__':
    main()
