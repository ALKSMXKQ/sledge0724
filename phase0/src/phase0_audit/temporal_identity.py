"""Phase 0C: real nuPlan identity, clock, coordinate, and representation transitions."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import gzip
import inspect
import json
import os
from pathlib import Path
import random
import traceback
import numpy as np
from .native_common import (CATEGORIES, TARGETS, REPO, config, dump, csv_write, provenance, parser, prepare,
                            load_raw, quantiles, sha)
from .scenario_index import database_index, connect
from .targeted_retention import require_gate
from .temporal_state import representation, cache_match, transitions, track_summary


def select(b3, b4, dbs, size, seed):
    rng = random.Random(seed)
    candidates = {r['path']:dict(r, selection_source='0B-4') for r in b4}
    candidates.update({r['path']:dict(r, selection_source='0B-3') for r in b3})
    available = [r for r in candidates.values() if r['log_name'] in dbs]
    selected = sorted([r for r in available if r['selection_source']=='0B-3'], key=lambda r:r['path'])
    rng.shuffle(selected)
    selected = selected[:size]
    used = {r['path'] for r in selected}
    for typ in TARGETS:
        options = sorted([r for r in available if r['scenario_type']==typ and r['path'] not in used],key=lambda r:r['path'])
        rng.shuffle(options)
        for r in options[:min(3, size-len(selected))]:
            selected.append(r); used.add(r['path'])
    options = sorted([r for r in available if r['path'] not in used],key=lambda r:r['path'])
    rng.shuffle(options)
    selected.extend(options[:size-len(selected)])
    if len(selected) != size:
        raise ValueError(f'Only {len(selected)} reusable scenarios have local databases; requested {size}')
    return selected, dict(candidate_count=len(candidates), with_local_db=len(available),
                          unavailable_local_db=len(candidates)-len(available),
                          by_type=dict(Counter(r['scenario_type'] for r in selected)),
                          by_source=dict(Counter(r['selection_source'] for r in selected)))


def make_scenario(scene, db, map_root, map_version, duration):
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    with connect(db) as conn:
        row = conn.execute('SELECT lp.timestamp, l.map_version FROM lidar_pc lp JOIN lidar ld ON ld.token=lp.lidar_token JOIN log l ON l.token=ld.log_token WHERE lp.token=?', (bytes.fromhex(scene['token']),)).fetchone()
    if row is None:
        raise ValueError(f'Anchor token missing from DB: {scene}')
    scenario = NuPlanScenario(data_root=str(db.parent), log_file_load_path=str(db),
        initial_lidar_token=scene['token'], initial_lidar_timestamp=row[0], scenario_type=scene['scenario_type'],
        map_root=str(map_root), map_version=map_version, map_name=row[1],
        scenario_extraction_info=ScenarioExtractionInfo(scenario_name=scene['scenario_type'], scenario_duration=duration,
                                                       extraction_offset=0., subsample_ratio=1.),
        ego_vehicle_parameters=get_pacifica_parameters())
    return scenario, row[0]


def api_inspection():
    from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    methods = ['get_number_of_iterations','get_time_point','get_ego_state_at_iteration','get_tracked_objects_at_iteration',
               'get_past_timestamps','get_future_timestamps','get_past_tracked_objects','get_future_tracked_objects',
               'get_ego_past_trajectory','get_ego_future_trajectory']
    return {cls.__name__:dict(source=inspect.getfile(cls), sha256=sha(inspect.getfile(cls)),
            methods={m:str(inspect.signature(getattr(cls,m))) for m in methods if hasattr(cls,m)})
            for cls in (AbstractScenario,NuPlanScenario,DetectionsTracks,TrackedObjects)}


def probe_api(scenario):
    result = {}
    for direction in ('past','future'):
        times = list(getattr(scenario, 'get_'+direction+'_timestamps')(0, time_horizon=1., num_samples=10))
        tracks = list(getattr(scenario, 'get_'+direction+'_tracked_objects')(0,time_horizon=1.,num_samples=10))
        egos = list(getattr(scenario,'get_ego_'+direction+'_trajectory')(0,time_horizon=1.,num_samples=10))
        aligned = len(times)==len(tracks)==len(egos)==10
        for time, detection, ego in zip(times,tracks,egos):
            aligned = aligned and ego.time_point.time_us == time.time_us
            aligned = aligned and all(o.metadata.timestamp_us == time.time_us for o in detection.tracked_objects.tracked_objects)
        result[direction] = dict(requested_samples=10,timestamps_us=[t.time_us for t in times],
                                  detections_count=[len(t.tracked_objects.tracked_objects) for t in tracks], aligned=aligned)
    return result


def db_link_checks(db, start, end):
    errors, count = [], 0
    with connect(db) as conn:
        sql = '''SELECT lb.token, lb.track_token, lb.prev_token, p.track_token, lb.next_token, n.track_token
                 FROM lidar_box lb JOIN lidar_pc lp ON lp.token=lb.lidar_pc_token
                 LEFT JOIN lidar_box p ON p.token=lb.prev_token LEFT JOIN lidar_box n ON n.token=lb.next_token
                 WHERE lp.timestamp BETWEEN ? AND ?'''
        for token, track, prev, prev_track, nex, next_track in conn.execute(sql, (start,end)):
            for label, linked, linked_track in [('prev',prev,prev_track),('next',nex,next_track)]:
                if linked is not None:
                    count += 1
                    if linked_track != track:
                        errors.append(dict(kind='database_box_link_identity_discontinuity', detection_token=token.hex(),
                                           track_token=track.hex(), direction=label, linked_token=linked.hex(),
                                           linked_track_token=linked_track.hex() if linked_track else None))
    return count, errors


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()), min=float(a.min()), max=float(a.max())) if len(a) else dict(n=0)


def run_scene(scene, db, args, cfg):
    from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_agent_feature import compute_ego_features
    scenario, anchor = make_scenario(scene, db, args.map_root, args.map_version, args.duration)
    n = scenario.get_number_of_iterations()
    if n < 2 or scenario.get_time_point(0).time_us != anchor:
        raise ValueError('Temporal sequence does not start at the cached anchor or has fewer than two frames')
    api = probe_api(scenario)
    if not all(r['aligned'] for r in api.values()):
        raise ValueError(f'Past/future API alignment failed: {api}')
    cached = load_raw(args.cache_root,scene)
    observations, per_frame, transforms, events, issues = [], [], [], [], []
    tracks = defaultdict(list)
    previous = None
    last_time = None
    timestamp_offsets = []
    with connect(db) as conn:
        for lidar_time, ego_time in conn.execute('SELECT lp.timestamp, ep.timestamp FROM lidar_pc lp JOIN ego_pose ep ON ep.token=lp.ego_pose_token WHERE lp.timestamp BETWEEN ? AND ?', (anchor,anchor+int(args.duration*1e6))):
            timestamp_offsets.append(ego_time-lidar_time)
    for i in range(n):
        time = scenario.get_time_point(i).time_us
        ego = scenario.get_ego_state_at_iteration(i)
        detection = scenario.get_tracked_objects_at_iteration(i)
        objects = detection.tracked_objects.tracked_objects
        if last_time is not None and time <= last_time:
            issues.append(dict(frame_index=i, kind='nonmonotonic_timestamp',timestamp_us=time))
        if ego.time_point.time_us != time:
            issues.append(dict(frame_index=i,kind='ego_timestamp_mismatch'))
        for field, vals in [('track_token',[o.track_token for o in objects]),('detection_token',[o.token for o in objects])]:
            c = Counter(vals)
            for token,count in c.items():
                if token is None or count>1:
                    issues.append(dict(frame_index=i,kind='duplicate_or_missing_'+field, token=token,count=count))
        statuses, raw_ids, elements, error, counts = representation(ego,detection,scenario.map_api,cfg)
        transforms.append(dict(frame_index=i,timestamp_us=time,by_category=error))
        if i==0:
            matches={cat:cache_match(getattr(cached,cat),elements[cat],raw_ids[cat]) for cat in CATEGORIES}
            ego_error=np.abs(cached.ego.states-compute_ego_features(ego).states).tolist()
            if any(not r['passed'] for r in matches.values()) or max(ego_error, default=0) >= 1e-5:
                issues.append(dict(frame_index=i,kind='cached_raw_correspondence_failure',matches=matches,ego_error=ego_error))
        frame = dict(frame_index=i,timestamp_us=time,dt_s=(time-last_time)/1e6 if last_time is not None else None,
            ego_rear_x=ego.rear_axle.x,ego_rear_y=ego.rear_axle.y,ego_heading=ego.rear_axle.heading,
            ego_center_x=ego.center.x,ego_center_y=ego.center.y,
            ego_vx=ego.dynamic_car_state.rear_axle_velocity_2d.x,ego_vy=ego.dynamic_car_state.rear_axle_velocity_2d.y,
            object_count=len(objects))
        for cat in CATEGORIES:
            for name in ('raw_count','in_frame_count','kept_count','dropped_by_cap'):
                frame[cat+'_'+name]=counts[cat][name]
        per_frame.append(frame)
        for obj in objects:
            if obj.metadata.timestamp_us != time:
                issues.append(dict(frame_index=i,kind='object_timestamp_mismatch',token=obj.token))
            state=statuses.get(obj.track_token)
            row=dict(frame_index=i,timestamp_us=time,track_token=obj.track_token,detection_token=obj.token,
                runtime_track_id=obj.metadata.track_id,actor_type=obj.tracked_object_type.name,
                global_x=obj.center.x,global_y=obj.center.y,global_heading=obj.center.heading,
                vx=obj.velocity.x if hasattr(obj,'velocity') else None,vy=obj.velocity.y if hasattr(obj,'velocity') else None,
                width=obj.box.width,length=obj.box.length,local_x=state['local'][0] if state else None,
                local_y=state['local'][1] if state else None,local_heading=state['local'][2] if state else None,
                speed=state['local'][5] if state and len(state['local'])==6 else None,
                representation_status=state['status'] if state else 'unsupported_actor_category')
            observations.append(row); tracks[obj.track_token].append(row)
        if previous is not None:
            events.extend(dict(frame_index=i,timestamp_us=time,**e) for e in transitions(previous,statuses))
        previous,status_before,last_time = statuses,previous,time
    track_rows = [track_summary(r) for r in tracks.values()]
    for r in track_rows:
        if r['unique_runtime_track_ids'] != 1 or r['category_changes']:
            issues.append(dict(kind='runtime_id_or_category_discontinuity',**r))
    linked_count, linked_errors = db_link_checks(db,per_frame[0]['timestamp_us'],per_frame[-1]['timestamp_us'])
    issues.extend(linked_errors)
    # Check vector coordinate conventions empirically against displacement; this is diagnostic, not a tracker.
    ego_errors_global, ego_errors_rotated, actor_velocity_errors = [], [], []
    for a,b in zip(per_frame,per_frame[1:]):
        dt=(b['timestamp_us']-a['timestamp_us'])/1e6
        observed=np.array([b['ego_rear_x']-a['ego_rear_x'],b['ego_rear_y']-a['ego_rear_y']])/dt
        v=np.array([a['ego_vx'],a['ego_vy']]); c,s=np.cos(a['ego_heading']),np.sin(a['ego_heading'])
        ego_errors_global.append(float(np.linalg.norm(observed-v)))
        ego_errors_rotated.append(float(np.linalg.norm(observed-np.array([c*v[0]-s*v[1],s*v[0]+c*v[1]]))))
    for rows in tracks.values():
        for a,b in zip(rows,rows[1:]):
            if b['frame_index']==a['frame_index']+1 and a['vx'] is not None:
                dt=(b['timestamp_us']-a['timestamp_us'])/1e6
                speed=np.array([b['global_x']-a['global_x'],b['global_y']-a['global_y']])/dt
                actor_velocity_errors.append(float(np.linalg.norm(speed-np.array([a['vx'],a['vy']]))))
    result=dict(token=scene['token'],scenario_type=scene['scenario_type'],frame_count=n,track_count=len(track_rows),
                observation_count=len(observations),nominal_database_interval_s=scenario.database_interval,
                dt_s=stats([f['dt_s'] for f in per_frame if f['dt_s'] is not None]),
                api_probe=api,db_link_checks=linked_count,cache_matches=matches,cache_ego_error=ego_error,
                ego_pose_minus_lidar_timestamp_us=stats(timestamp_offsets),
                ego_velocity_global_frame_residual_mps=stats(ego_errors_global),
                ego_velocity_rotated_from_local_residual_mps=stats(ego_errors_rotated),
                actor_velocity_global_frame_residual_mps=stats(actor_velocity_errors),
                transition_counts=dict(Counter(e['event'] for e in events)),issue_count=len(issues))
    return result,per_frame,track_rows,observations,transforms,events,issues


def main():
    p=parser(__doc__,'temporal_identity',30)
    p.add_argument('--db-root',type=Path,required=True)
    p.add_argument('--map-root',type=Path,default=Path(os.environ['NUPLAN_MAPS_ROOT']) if os.getenv('NUPLAN_MAPS_ROOT') else None)
    p.add_argument('--map-version',default='nuplan-maps-v1.0',help='Installed nuPlan scenario-builder YAML default')
    p.add_argument('--duration',type=float,default=5.,help='Contiguous future rollout duration in seconds at native DB rate')
    p.add_argument('--equivalence-dir',type=Path,default=REPO/'phase0/outputs/processor_equivalence')
    p.add_argument('--targeted-dir',type=Path,default=REPO/'phase0/outputs/targeted_retention')
    args=p.parse_args(); cfg=config()
    require_gate(args.equivalence_dir/'summary.json',cfg)
    if json.loads((args.targeted_dir/'summary.json').read_text())['status']!='COMPLETE':
        raise ValueError('0B-4 must be COMPLETE before 0C')
    if args.map_root is None or not args.map_root.is_dir() or args.duration<=0:
        raise ValueError('Existing --map-root and positive --duration are required')
    prepare(args)
    dump(args.output_dir/'environment.json',provenance(cfg))
    dump(args.output_dir/'api_inspection.json',api_inspection())
    dbs=database_index(args.db_root)
    selected,selection=select(json.loads((args.equivalence_dir/'selection.json').read_text())['selected'],
        json.loads((args.targeted_dir/'selected_scenes.json').read_text()),dbs,args.sample_size,args.seed)
    dump(args.output_dir/'selection.json',dict(seed=args.seed,selected=selected,**selection))
    (args.output_dir/'selected_tokens.txt').write_text(''.join(r['token']+'\n' for r in selected))
    results,frames,tracks,checks,events,failures=[],[],[],[],[],[]
    with gzip.open(args.output_dir/'actor_observations.csv.gz','wt',newline='') as f:
        writer=None
        for i,scene in enumerate(selected):
            meta={k:scene[k] for k in ('token','log_name','scenario_type')}
            try:
                result,fr,tr,obs,ch,ev,fail=run_scene(scene,dbs[scene['log_name']],args,cfg)
                results.append(result)
                frames.extend(dict(**meta,**r) for r in fr)
                tracks.extend(dict(**meta,**r) for r in tr)
                checks.append(dict(**meta,frames=ch,cache_matches=result['cache_matches'],cache_ego_error=result['cache_ego_error']))
                events.extend(dict(**meta,**r) for r in ev)
                failures.extend(dict(**meta,**r) for r in fail)
                for r in obs:
                    row=dict(**meta,**r)
                    if writer is None:
                        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row)
            except Exception as exc:
                failures.append(dict(**meta,kind='scene_execution_failure',error=repr(exc),traceback=traceback.format_exc()))
            dump(args.output_dir/'identity_failures.json',failures)
            print(f'Temporal scene {i+1}/{len(selected)}; completed={len(results)}; issues={len(failures)}',flush=True)
    csv_write(args.output_dir/'per_frame.csv',frames)
    csv_write(args.output_dir/'per_track.csv',tracks)
    csv_write(args.output_dir/'transitions.csv',events)
    dump(args.output_dir/'transform_checks.json',checks)
    dump(args.output_dir/'per_scenario.json',results)
    dt=[r['dt_s'] for r in frames if r['dt_s'] is not None]
    transform_max=[x for check in checks for frame in check['frames'] for err in frame['by_category'].values() for x in err['max_abs_error']]
    event_counts=dict(Counter(r['event'] for r in events))
    summary=dict(status='PASS' if len(results)==args.sample_size and not failures else 'FAIL',
        sample_count_selected=len(selected),sample_count_completed=len(results),frame_count=len(frames),track_count=len(tracks),
        total_actor_observations=sum(r['observed_frames'] for r in tracks),
        tracks_observed_in_multiple_frames=sum(r['observed_frames']>1 for r in tracks),
        tracks_with_observation_gaps=sum(r['observation_gap_count']>0 for r in tracks),
        unique_runtime_track_id_failures=sum(r['unique_runtime_track_ids']!=1 for r in tracks),
        detection_token_changes=sum(r['detection_token_changes'] for r in tracks),
        max_consecutive_frames=quantiles([r['max_consecutive_frames'] for r in tracks]),
        database_link_checks=sum(r['db_link_checks'] for r in results),identity_failures=len(failures),
        dt_s=stats(dt),max_coordinate_transform_error=max(transform_max,default=None),
        transition_counts=event_counts,selection=selection,provenance=provenance(cfg),
        limitations=['Identity checks validate annotated identities and DB links; undetected physical ID swaps cannot be ruled out.',
                     'Observation disappearance and reappearance are not automatically spatial exits or ID changes.',
                     'Actual timestamps must be used; finite time resolution does not establish hazard event timing precision.',
                     'Runtime track_id is stable only within a process; persist log_name plus track_token.',
                     'Five-second forward rollouts plus separate past/future API probes are not a full-dataset audit.'])
    dump(args.output_dir/'summary.json',summary)
    print({k:v for k,v in summary.items() if k not in ('provenance','limitations')},flush=True)
    if summary['status']!='PASS':
        raise SystemExit(2)

if __name__=='__main__':
    main()
