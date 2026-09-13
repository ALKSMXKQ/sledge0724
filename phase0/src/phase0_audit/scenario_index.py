"""Resolve native cache metadata against read-only nuPlan scenario_tag tables."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import sqlite3


def database_index(root):
    index = {}
    for path in sorted(Path(root).rglob('*.db')):
        if path.stem in index:
            raise ValueError(f'Ambiguous database basename: {path.stem}; choose a single split root')
        index[path.stem] = path
    if not index:
        raise ValueError(f'No nuPlan databases under {root}')
    return index


def connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)


def classify_unknown(tags, token_exists):
    if not token_exists:
        return 'unknown', 'token_missing_in_local_db'
    if len(tags) == 1:
        return tags[0], 'resolved_single_authoritative_tag'
    if len(tags) > 1:
        return 'unknown', 'multiple_tags_ambiguous'
    return 'unknown', 'unlabeled_in_authoritative_db'


def resolve(entries, dbs):
    by_log = defaultdict(list)
    for r in entries:
        by_log[r['log_name']].append(r)
    resolved, unknown_rows, issues, log_rows = [], [], [], []
    for i, (log, rows) in enumerate(sorted(by_log.items())):
        tags = defaultdict(list)
        tokens = set()
        if log in dbs:
            with connect(dbs[log]) as conn:
                for token, typ in conn.execute('SELECT lidar_pc_token, type FROM scenario_tag'):
                    tags[token.hex()].append(typ)
                tokens = {r[0].hex() for r in conn.execute('SELECT token FROM lidar_pc')}
        counts = Counter()
        for row in rows:
            typ = row['scenario_type']
            original = typ
            authoritative = sorted(set(tags[row['token']]))
            source = 'native_cache_directory'
            if typ == 'unknown':
                if log not in dbs:
                    reason = 'database_unavailable'
                else:
                    typ, reason = classify_unknown(authoritative, row['token'] in tokens)
                source = 'nuplan_scenario_tag' if typ != 'unknown' else 'unresolved_'+reason
                counts[reason] += 1
                unknown_rows.append(dict(log_name=log, token=row['token'], path=row['path'],
                    original_type=original, resolved_type=typ, reason=reason, authoritative_tags='|'.join(authoritative)))
            elif log in dbs and typ not in authoritative:
                issues.append(dict(**row, authoritative_tags=authoritative,
                                   reason='cache_type_not_in_local_db_tags'))
            resolved.append(dict(**row, original_scenario_type=original, resolved_scenario_type=typ,
                                 type_source=source, database_available=log in dbs))
        log_rows.append(dict(log_name=log, cache_scenes=len(rows), unknown_count=sum(counts.values()),
                            resolution_counts=dict(counts), database_available=log in dbs))
        if (i+1) % 100 == 0:
            print(f'Metadata resolution {i+1}/{len(by_log)} logs', flush=True)
    return resolved, unknown_rows, issues, log_rows
