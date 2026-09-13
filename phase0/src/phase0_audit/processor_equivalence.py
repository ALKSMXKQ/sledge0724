"""Phase 0B-3: validate retention replay against the complete native processor."""
from __future__ import annotations
import random
import numpy as np
from .native_common import (CATEGORIES, TARGETS, DIMS, config, dump, csv_write, provenance,
                            parser, prepare, inventory, load_raw, replay, balanced_sample, sha)
from .retention_audit import analyze_actor_element


def compare(raw, processed, category, cfg):
    expected = replay(getattr(raw, category).states, category, cfg)
    actual = getattr(processed, category)
    e, a = expected['states'], actual.states
    shape_ok = e.shape == a.shape
    error = np.abs(e.astype(float)-a.astype(float)) if shape_ok else np.full(e.shape, 1e30)
    count = int(np.sum(actual.mask))
    from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_utils import coords_in_frame
    raw_states = getattr(raw, category).states
    native_inside = coords_in_frame(raw_states[:, :2], cfg.frame) if len(raw_states) else np.zeros(0, bool)
    state_ok = shape_ok and bool(np.isfinite(a).all()) and bool(np.all(error < 1e-5))
    # Output has no IDs. Ordered state comparison is observable; identical states are indistinguishable.
    ordering_ok = state_ok
    mask_ok = np.array_equal(expected['mask'], actual.mask) and actual.mask.dtype == np.bool_
    speed_ok = True if category == 'static_objects' else shape_ok and np.array_equal(e[:, 5], a[:, 5])
    legacy = analyze_actor_element({category: {'states': getattr(raw, category).states,
                                              'mask': getattr(raw, category).mask}}, category,
                                  getattr(cfg, 'num_'+category))
    legacy_ok = all(legacy[k] == expected['metrics'][k] for k in
                    ('raw_count', 'in_frame_count', 'kept_count', 'dropped_outside_frame', 'dropped_by_cap'))
    row = dict(actor_category=category, **expected['metrics'], actual_processed_valid_count=count,
               native_in_frame_count=int(native_inside.sum()), in_frame_agreement=np.array_equal(expected['inside'], native_inside),
               count_agreement=count == expected['metrics']['kept_count'], ordering_agreement=ordering_ok,
               mask_agreement=mask_ok, speed_clipping_agreement=bool(speed_ok), legacy_count_agreement=legacy_ok,
               raw_dtype=str(getattr(raw, category).states.dtype), predicted_dtype=str(e.dtype),
               processed_dtype=str(a.dtype), max_abs_error=float(error.max()), mean_abs_error=float(error.mean()))
    for j, dim in enumerate(DIMS):
        row[dim+'_error'] = float(error[:, j].max()) if j < error.shape[1] else None
        row[dim+'_mean_error'] = float(error[:, j].mean()) if j < error.shape[1] else None
    detail = dict(expected_states=e.tolist(), actual_states=a.tolist(), expected_mask=expected['mask'].tolist(),
                  actual_mask=actual.mask.tolist(), selected_raw_indices=expected['selected'].tolist(),
                  raw_values=getattr(raw, category).states.tolist(),
                  raw_speed=None if expected['raw_speeds'] is None else expected['raw_speeds'].tolist(),
                  predicted_clipped_speed=e[expected['mask'], 5].tolist() if e.shape[1] == 6 else None,
                  native_processed_speed=a[actual.mask, 5].tolist() if a.shape[1] == 6 else None)
    passed = all(row[k] for k in ('count_agreement', 'ordering_agreement', 'mask_agreement',
                                 'speed_clipping_agreement', 'legacy_count_agreement', 'in_frame_agreement')) and state_ok
    mismatch = None if passed else dict(**row, **detail,
        state_index=np.argwhere(error >= 1e-5).tolist(),
        possible_source_code_reason='Inspect native frame predicate, default np.argsort ties, float32 assignment before np.minimum, and dummy raw mask semantics.')
    return row, detail, mismatch


def select(root, entries, cfg, size, seed, pool_size):
    rng = random.Random(seed)
    pool = rng.sample(entries, min(pool_size, len(entries)))
    seen = {r['path'] for r in pool}
    for typ in TARGETS:
        for r in balanced_sample([r for r in entries if r['scenario_type'] == typ], 40, seed):
            if r['path'] not in seen:
                pool.append(r)
                seen.add(r['path'])
    candidates = []
    for i, row in enumerate(pool):
        raw = load_raw(root, row)
        metrics = {cat: replay(getattr(raw, cat).states, cat, cfg)['metrics'] for cat in CATEGORIES}
        reasons = []
        if all(m['dropped_by_cap'] == 0 for m in metrics.values()):
            reasons.append('normal_no_overflow')
        for cat in ('pedestrians', 'static_objects'):
            if metrics[cat]['dropped_by_cap']:
                reasons.append(cat+'_overflow')
        if row['scenario_type'] in TARGETS:
            reasons.append(row['scenario_type'])
        candidates.append(dict(**row, selection_reasons=reasons, candidate_metrics=metrics))
        if (i+1) % 200 == 0:
            print(f'Candidate screening {i+1}/{len(pool)}', flush=True)
    strata = ['normal_no_overflow', 'pedestrians_overflow', 'static_objects_overflow', *TARGETS]
    availability = {s: sum(s in r['selection_reasons'] for r in candidates) for s in strata}
    selected, used = [], set()
    quota = max(1, size//len(strata))
    for s in strata:
        options = [r for r in candidates if s in r['selection_reasons'] and r['path'] not in used]
        rng.shuffle(options)
        for r in options[:min(quota, size-len(selected))]:
            selected.append(dict(**r, primary_selection_reason=s))
            used.add(r['path'])
    other = [r for r in candidates if r['path'] not in used]
    rng.shuffle(other)
    selected.extend(dict(**r, primary_selection_reason='seeded_fill') for r in other[:size-len(selected)])
    return selected, dict(candidate_pool_size=len(pool), strata_available_in_pool=availability,
                          scope='Overflow availability measured in candidate pool; scenario type availability from full inventory',
                          full_type_counts={s: sum(r['scenario_type'] == s for r in entries) for s in TARGETS},
                          candidates=candidates)


def main():
    p = parser(__doc__, 'processor_equivalence', 100)
    p.add_argument('--candidate-pool-size', type=int, default=1000)
    args = p.parse_args()
    prepare(args)
    cfg = config()
    dump(args.output_dir/'environment.json', provenance(cfg))
    entries = inventory(args.cache_root)
    selected, selection = select(args.cache_root, entries, cfg, args.sample_size, args.seed, args.candidate_pool_size)
    dump(args.output_dir/'selection.json', dict(seed=args.seed, inventory_count=len(entries), selected=selected, **selection))
    (args.output_dir/'selected_tokens.txt').write_text(''.join(r['token']+'\n' for r in selected))
    rows, details, mismatches = [], [], []
    from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import sledge_raw_feature_processing
    for i, scene in enumerate(selected):
        meta = {k: scene[k] for k in ('token', 'scenario_type', 'log_name', 'path')}
        try:
            raw = load_raw(args.cache_root, scene)
            processed, raster = sledge_raw_feature_processing(raw, cfg)
            for cat in CATEGORIES:
                row, detail, mismatch = compare(raw, processed, cat, cfg)
                rows.append(dict(**meta, **row))
                details.append(dict(**meta, actor_category=cat, cache_sha256=sha(args.cache_root/scene['path']), **detail))
                if mismatch:
                    mismatches.append(dict(**meta, **mismatch))
        except Exception as exc:
            mismatches.append(dict(**meta, exception=repr(exc), possible_source_code_reason='Native load/processor exception; not removed from selected sample.'))
        print(f'Native equivalence {i+1}/{len(selected)}; mismatches={len(mismatches)}', flush=True)
    fields = ('count_agreement', 'ordering_agreement', 'mask_agreement', 'speed_clipping_agreement', 'legacy_count_agreement', 'in_frame_agreement')
    summary = dict(status='PASS' if len(rows) == args.sample_size*3 and not mismatches else 'FAIL',
                   sample_count=len(selected), actor_category_checks=len(rows), seed=args.seed,
                   inventory_count=len(entries), max_abs_error=max((r['max_abs_error'] for r in rows), default=None),
                   tolerance_strict_less_than=1e-5, mismatch_count=len(mismatches),
                   **{f: sum(r[f] for r in rows)/(len(selected)*3) for f in fields},
                   ordering_definition='Full ordered state rows; identical states have no distinguishable ID in processed output',
                   predicted_in_frame_note='Native processor exposes post-cap state; pre-cap mask/count also compared with native coords_in_frame for every real scene.',
                   provenance=provenance(cfg))
    summary['allows_0b4'] = summary['status'] == 'PASS'
    dump(args.output_dir/'summary.json', summary)
    csv_write(args.output_dir/'per_scene.csv', rows)
    dump(args.output_dir/'state_comparisons.json', details)
    dump(args.output_dir/'mismatches.json', mismatches)
    print({k:v for k,v in summary.items() if k != 'provenance'}, flush=True)
    if not summary['allows_0b4']:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
