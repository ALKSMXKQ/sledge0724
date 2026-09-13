"""Object-level temporal diagnostics; no hazard semantics or tracker inference."""
from __future__ import annotations
from collections import Counter
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .native_common import CATEGORIES, replay


def local_state(obj, ego):
    """Independent scalar SE(2) transform around ego CENTER, matching SLEDGE's origin."""
    dx, dy = obj.center.x-ego.center.x, obj.center.y-ego.center.y
    c, s = math.cos(ego.center.heading), math.sin(ego.center.heading)
    theta = obj.center.heading-ego.center.heading
    result = [c*dx+s*dy, -s*dx+c*dy, math.atan2(math.sin(theta), math.cos(theta)), obj.box.width, obj.box.length]
    if hasattr(obj, 'velocity'):
        result.append(math.hypot(obj.velocity.x, obj.velocity.y))
    return result


def representation(ego, detections, map_api, cfg):
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from sledge.autoencoder.preprocessing.feature_builders.sledge_raw_feature_builder import get_drivable_area_map
    from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_agent_feature import compute_agent_features, compute_static_object_features
    from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import process_agents, process_static_objects
    area = get_drivable_area_map(map_api, ego, cfg.radius)
    groups = dict(vehicles=detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE),
                  pedestrians=detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.PEDESTRIAN),
                  static_objects=detections.tracked_objects.get_static_objects())
    statuses, raw_ids, elements, errors, counts = {}, {}, {}, {}, {}
    for cat, objects in groups.items():
        dim = 5 if cat == 'static_objects' else 6
        a = np.asarray([local_state(obj, ego)[:dim] for obj in objects], dtype=float).reshape(-1, dim)
        on_area = area.points_in_polygons(np.asarray([[obj.center.x,obj.center.y] for obj in objects])).any(axis=0) if cat == 'vehicles' and objects else np.ones(len(objects), bool)
        within_radius = np.linalg.norm(a[:, :2], axis=1) <= cfg.radius
        valid = on_area & within_radius
        expected_raw = a[valid].astype(np.float32)
        if cat == 'static_objects':
            native_raw = compute_static_object_features(ego, detections, cfg.radius)
            native_processed, _ = process_static_objects(native_raw, cfg)
        else:
            typ = TrackedObjectType.VEHICLE if cat == 'vehicles' else TrackedObjectType.PEDESTRIAN
            native_raw = compute_agent_features(ego, detections, typ, cfg.radius, area if cat == 'vehicles' else None)
            limit = cfg.vehicle_max_velocity if cat == 'vehicles' else cfg.pedestrian_max_velocity
            native_processed, _ = process_agents(native_raw, cfg, getattr(cfg, 'num_'+cat), limit)
        native_array = native_raw.states.reshape(-1, dim)
        if expected_raw.shape != native_array.shape:
            raise ValueError(f'Independent/native raw count mismatch {cat}: {expected_raw.shape}, {native_array.shape}')
        error = np.abs(a[valid]-native_array.astype(float))
        errors[cat] = dict(count=len(native_array), max_abs_error=error.max(axis=0).tolist() if len(error) else [0.]*dim,
                           mean_abs_error=error.mean(axis=0).tolist() if len(error) else [0.]*dim,
                           raw_dtype=str(native_array.dtype), independent_dtype=str(a.dtype))
        if error.size and error.max() >= 1e-5:
            raise ValueError(f'Independent/native transform mismatch {cat}: {error.max()}')
        result = replay(native_array, cat, cfg)
        if not np.array_equal(result['states'], native_processed.states) or not np.array_equal(result['mask'], native_processed.mask):
            raise ValueError(f'Temporal native processing differs from validated replay: {cat}')
        original_indices = np.flatnonzero(valid)
        raw_ids[cat] = [objects[i].track_token for i in original_indices]
        for i, obj in enumerate(objects):
            state = 'raw_drivable_area_exclusion' if not on_area[i] else 'raw_radius_exclusion' if not within_radius[i] else 'outside_frame'
            statuses[obj.track_token] = dict(category=cat, local=a[i].tolist(), status=state,
                                            inside_frame=bool(np.all(np.abs(a[i,:2].astype(np.float32)) <= np.asarray(cfg.frame)/2)))
        selected = set(result['selected'].tolist())
        for raw_i, obj_i in enumerate(original_indices):
            if result['inside'][raw_i]:
                statuses[objects[obj_i].track_token]['status'] = 'kept' if raw_i in selected else 'cap_excluded'
        elements[cat] = native_raw
        counts[cat] = result['metrics']
    return statuses, raw_ids, elements, errors, counts


def cache_match(cached, rebuilt, ids, tolerance=1e-5):
    """State matching is only for raw-cache correspondence, never to reorder B3 outputs."""
    dim = rebuilt.states.shape[-1] if rebuilt.states.ndim == 2 else (cached.states.shape[-1] if cached.states.ndim == 2 else 6)
    a = cached.states.reshape(-1, dim)
    b = rebuilt.states.reshape(-1, dim)
    if len(a) != len(b):
        return dict(passed=False, reason='cache_vs_current_raw_count_difference', cache_count=len(a), rebuilt_count=len(b), matches=[])
    if not len(a):
        return dict(passed=True, cache_count=0, rebuilt_count=0, max_abs_error=0., ambiguous_count=0, matches=[])
    costs = np.max(np.abs(a[:,None,:].astype(float)-b[None,:,:].astype(float)),axis=2)
    left, right = linear_sum_assignment(costs)
    matches = [dict(cache_index=int(i), rebuilt_index=int(j), track_token=ids[j],
                    max_abs_error=float(costs[i,j]), state_error=np.abs(a[i].astype(float)-b[j]).tolist(),
                    cached_state=a[i].tolist(), rebuilt_state=b[j].tolist(),
                    ambiguous=bool(np.sum(costs[i] < tolerance)>1)) for i,j in zip(left,right)]
    return dict(passed=bool(np.all(costs[left,right] < tolerance)), cache_count=len(a), rebuilt_count=len(b),
                max_abs_error=float(costs[left,right].max()), ambiguous_count=sum(r['ambiguous'] for r in matches), matches=matches)


def transitions(previous, current):
    events = []
    for token in sorted(set(previous) | set(current)):
        old, new = previous.get(token), current.get(token)
        if old is None:
            event = 'observation_appearance_censored'
        elif new is None:
            event = 'observation_disappearance_unknown_cause'
        elif old['inside_frame'] != new['inside_frame']:
            event = 'spatial_entry' if new['inside_frame'] else 'spatial_exit'
        elif old['status'] == 'kept' and new['status'] == 'cap_excluded':
            event = 'capacity_exclusion'
        elif old['status'] == 'cap_excluded' and new['status'] == 'kept':
            event = 'capacity_readmission'
        elif old['status'] != new['status']:
            event = 'other_representation_transition'
        else:
            continue
        events.append(dict(track_token=token, event=event,
                           previous_status=old['status'] if old else None, current_status=new['status'] if new else None))
    return events


def track_summary(rows):
    frames = [r['frame_index'] for r in rows]
    spans, current = [], 1
    for a,b in zip(frames, frames[1:]):
        if b == a+1:
            current += 1
        else:
            spans.append(current)
            current = 1
    spans.append(current)
    return dict(track_token=rows[0]['track_token'], actor_type=rows[0]['actor_type'],
                first_frame=frames[0], last_frame=frames[-1], observed_frames=len(rows),
                max_consecutive_frames=max(spans), observation_gap_count=len(spans)-1,
                unique_detection_tokens=len({r['detection_token'] for r in rows}),
                detection_token_changes=sum(a['detection_token'] != b['detection_token'] for a,b in zip(rows,rows[1:])),
                unique_runtime_track_ids=len({r['runtime_track_id'] for r in rows}),
                category_changes=len({r['actor_type'] for r in rows})-1,
                first_timestamp_us=rows[0]['timestamp_us'], last_timestamp_us=rows[-1]['timestamp_us'],
                kept_frames=sum(r['representation_status']=='kept' for r in rows),
                cap_excluded_frames=sum(r['representation_status']=='cap_excluded' for r in rows),
                outside_frame_frames=sum(r['representation_status']=='outside_frame' for r in rows))
