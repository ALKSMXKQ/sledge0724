"""Phase 0B-4: target-type retention and explicit unknown-type accounting."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
from .native_common import (CATEGORIES, TARGETS, config, dump, csv_write, provenance, parser, prepare,
                            inventory, load_raw, replay, balanced_sample, quantiles, sha, REPO)
from .scenario_index import database_index, resolve


def require_gate(path, cfg):
    gate = json.loads(path.read_text())
    current = provenance(cfg)
    if gate.get('status') != 'PASS' or not gate.get('allows_0b4'):
        raise ValueError('0B-3 PASS is required before 0B-4')
    for key in ('config_sha256', 'processor_sha256', 'replay_sha256', 'packages'):
        if gate['provenance'][key] != current[key]:
            raise ValueError(f'0B-3 provenance changed: {key}; rerun equivalence')
    return gate


def aggregate(rows):
    inside = sum(r['in_frame_count'] for r in rows)
    kept = sum(r['kept_count'] for r in rows)
    return dict(sample_count=len(rows), nonempty_scene_count=sum(r['raw_count'] > 0 for r in rows),
                in_frame_nonempty_scene_count=sum(r['in_frame_count'] > 0 for r in rows),
                scene_fraction_with_cap_drop=sum(r['dropped_by_cap'] > 0 for r in rows)/len(rows) if rows else None,
                total_raw=sum(r['raw_count'] for r in rows), total_in_frame=inside, total_kept=kept,
                total_cap_dropped=sum(r['dropped_by_cap'] for r in rows),
                total_spatial_crop_dropped=sum(r['dropped_outside_frame'] for r in rows),
                p_kept_given_inside=kept/inside if inside else None,
                overflow=quantiles([r['overflow'] for r in rows]),
                positive_overflow=quantiles([r['overflow'] for r in rows if r['overflow'] > 0]),
                d_K=quantiles([r['d_K'] for r in rows]),
                d_first_drop=quantiles([r['d_first_drop'] for r in rows]),
                boundary_gap=quantiles([r['boundary_gap'] for r in rows]))


def main():
    p = parser(__doc__, 'targeted_retention', 150)
    p.add_argument('--db-root', type=Path, required=True)
    p.add_argument('--equivalence-summary', type=Path, default=REPO/'phase0/outputs/processor_equivalence/summary.json')
    args = p.parse_args()
    cfg = config()
    require_gate(args.equivalence_summary, cfg)
    prepare(args)
    dump(args.output_dir/'environment.json', provenance(cfg))
    entries = inventory(args.cache_root)
    resolved, unknown, issues, logs = resolve(entries, database_index(args.db_root))
    csv_write(args.output_dir/'unknown_resolution_per_token.csv', unknown)
    unknown_summary = dict(inventory_count=len(entries), originally_unknown=len(unknown),
        resolved_unknown=sum(r['resolved_type'] != 'unknown' for r in unknown),
        unresolved_unknown=sum(r['resolved_type'] == 'unknown' for r in unknown),
        reasons=dict(Counter(r['reason'] for r in unknown)),
        source='Native cache directory schema, checked against nuPlan SQLite scenario_tag and lidar_pc',
        upstream_reason='get_scenarios_from_db LEFT OUTER JOIN scenario_tag; null scenario_type becomes DEFAULT_SCENARIO_NAME=unknown in scenario filter utils',
        cache_db_type_disagreements=issues, logs=logs,
        historical_1000_note='Original summary records 295 unknown but lacks its full token list. This run accounts for every currently cached unknown token, without claiming to recover the exact historical 295.')
    dump(args.output_dir/'unknown_resolution.json', unknown_summary)
    selections, counts = [], []
    for typ in TARGETS:
        eligible = [dict(r, scenario_type=r['resolved_scenario_type']) for r in resolved if r['resolved_scenario_type'] == typ]
        chosen = balanced_sample(eligible, args.sample_size, args.seed)
        selections.extend(chosen)
        counts.append(dict(scenario_type=typ, available_count=len(eligible), sample_count=len(chosen),
                           sampling='all' if len(eligible)<=args.sample_size else 'equal_allocation_by_log_then_seeded_within_log',
                           distinct_selected_logs=len({r['log_name'] for r in chosen})))
    csv_write(args.output_dir/'scenario_type_counts.csv', counts)
    dump(args.output_dir/'selected_scenes.json', selections)
    (args.output_dir/'selected_tokens.txt').write_text(''.join(r['token']+'\n' for r in selections))
    rows = []
    for i, scene in enumerate(selections):
        raw = load_raw(args.cache_root, scene)
        for cat in CATEGORIES:
            result = replay(getattr(raw, cat).states, cat, cfg)
            rows.append(dict(token=scene['token'], scenario_type=scene['scenario_type'], log_name=scene['log_name'],
                path=scene['path'], actor_category=cat, type_source=scene['type_source'], **result['metrics']))
        if (i+1)%100 == 0:
            print(f'Targeted retention {i+1}/{len(selections)}', flush=True)
    csv_write(args.output_dir/'per_scene.csv', rows)
    summary = dict(status='COMPLETE' if all(r['sample_count'] > 0 for r in counts) else 'INCOMPLETE',
                   sample_count=len(selections), seed=args.seed, per_type_requested=args.sample_size,
                   sampling_estimand='Log-balanced selected target scenes; not a scene-uniform estimate of the entire cache',
                   unknown_in_targeted_statistics=0, unknown_resolution={k:v for k,v in unknown_summary.items() if k not in ('logs','cache_db_type_disagreements')},
                   gate_0b3_summary_sha256=sha(args.equivalence_summary), provenance=provenance(cfg),
                   by_scenario_type={typ:{cat:aggregate([r for r in rows if r['scenario_type']==typ and r['actor_category']==cat]) for cat in CATEGORIES} for typ in TARGETS})
    dump(args.output_dir/'summary.json', summary)
    examples = {typ:{cat:sorted([r for r in rows if r['scenario_type']==typ and r['actor_category']==cat and r['d_first_drop'] is not None],
                               key=lambda r:r['d_first_drop'])[:5] for cat in CATEGORIES} for typ in TARGETS}
    dump(args.output_dir/'examples.json', examples)
    print(dict(status=summary['status'], sample_count=len(selections), unknown_reasons=unknown_summary['reasons'],
               unknown_remaining=unknown_summary['unresolved_unknown'], metadata_issues=len(issues)), flush=True)
    if summary['status'] != 'COMPLETE':
        raise SystemExit(2)

if __name__ == '__main__':
    main()
