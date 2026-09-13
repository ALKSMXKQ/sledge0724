"""Read-only, native-config-backed measurement utilities for Phase 0."""
from __future__ import annotations
import argparse
import csv
import dataclasses
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import sys
import numpy as np

CATEGORIES = ('vehicles', 'pedestrians', 'static_objects')
TARGETS = ('near_pedestrian_on_crosswalk', 'traversing_crosswalk', 'on_pickup_dropoff',
           'traversing_pickup_dropoff', 'on_traffic_light_intersection', 'traversing_intersection')
DIMS = ('x', 'y', 'heading', 'width', 'length', 'speed')
REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / 'sledge/script/config/common/autoencoder_model/rvae_model.yaml'


def config():
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    return instantiate(OmegaConf.load(CONFIG).config, _convert_='all')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(cfg):
    from sledge.autoencoder.preprocessing.feature_builders.sledge import sledge_feature_processing as native
    versions = {}
    for name in ('numpy', 'torch', 'opencv-python', 'hydra-core', 'nuplan-devkit', 'scipy', 'shapely'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed as distribution'
    import nuplan
    return dict(python=sys.executable, conda_env=Path(sys.executable).parents[1].name,
                packages=versions, nuplan_source=nuplan.__file__, SLEDGE_EXP_ROOT=os.getenv('SLEDGE_EXP_ROOT'),
                command=[sys.executable, *sys.argv], config_source=str(CONFIG.relative_to(REPO)),
                config_sha256=sha(CONFIG), config=dataclasses.asdict(cfg),
                processor_source=native.__file__, processor_sha256=sha(native.__file__),
                replay_sha256=sha(__file__))


def parser(description, output, size):
    p = argparse.ArgumentParser(description=description)
    exp = os.getenv('SLEDGE_EXP_ROOT')
    p.add_argument('--cache-root', type=Path, default=Path(exp)/'caches/autoencoder_cache' if exp else None)
    p.add_argument('--output-dir', type=Path, default=REPO/'phase0/outputs'/output)
    p.add_argument('--sample-size', type=int, default=size)
    p.add_argument('--seed', type=int, default=9102026)
    return p


def prepare(args):
    if args.cache_root is None or not args.cache_root.is_dir():
        raise ValueError('Pass an existing --cache-root or set SLEDGE_EXP_ROOT')
    if args.sample_size <= 0:
        raise ValueError('--sample-size must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing audit: {args.output_dir}')


def inventory(root):
    """Exact native cache schema: log/scenario_type/token/sledge_raw.gz."""
    rows = []
    for log in sorted(root.iterdir()):
        if not log.is_dir() or log.name == 'metadata':
            continue
        for typ in sorted(log.iterdir()):
            if not typ.is_dir():
                continue
            for token in sorted(typ.iterdir()):
                path = token/'sledge_raw.gz'
                if path.is_file():
                    rows.append(dict(log_name=log.name, scenario_type=typ.name, token=token.name,
                                     path=str(path.relative_to(root))))
    if not rows:
        raise ValueError('No native cache entries found')
    return rows


def load_raw(root, row):
    with gzip.open(root/row['path'], 'rb') as f:
        data = pickle.load(f)
    from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorRaw, SledgeVectorElement
    # Inherited deserialize returns SledgeVector, not SledgeVectorRaw in this checkout.
    return SledgeVectorRaw(**{k: SledgeVectorElement.deserialize(v) for k, v in data.items()})


def replay(states, category, cfg):
    """Independent replay; raw mask is deliberately not an actor-validity mask."""
    width = 5 if category == 'static_objects' else 6
    a = np.asarray(states)
    if a.size == 0:
        a = a.reshape(0, width)
    if a.ndim != 2 or a.shape[1] != width or not np.isfinite(a).all():
        raise ValueError(f'Invalid {category} state {a.shape}')
    cap = getattr(cfg, 'num_' + category)
    bounds = np.asarray(cfg.frame)/2
    inside = ((a[:, :2] >= -bounds) & (a[:, :2] <= bounds)).all(axis=1)
    indices = np.flatnonzero(inside)
    distances = np.linalg.norm(a[inside, :2], axis=-1)
    order = np.argsort(distances)  # Native default quicksort; no secondary key.
    selected = indices[order[:cap]]
    dropped = indices[order[cap:]]
    predicted = np.zeros((cap, width), dtype=np.float32)
    predicted[:len(selected)] = a[selected]
    raw_speeds = a[selected, 5].copy() if width == 6 else None
    if width == 6:
        limit = getattr(cfg, 'vehicle_max_velocity' if category == 'vehicles' else 'pedestrian_max_velocity')
        predicted[:, 5] = np.minimum(predicted[:, 5], limit)
    mask = np.arange(cap) < len(selected)
    dk = float(distances[order[len(selected)-1]]) if len(selected) else None
    first = float(distances[order[cap]]) if len(dropped) else None
    metrics = dict(raw_count=len(a), in_frame_count=len(indices), kept_count=len(selected),
                   dropped_outside_frame=int((~inside).sum()), dropped_by_cap=len(dropped),
                   cap=cap, overflow=len(indices)-cap, d_K=dk, d_first_drop=first,
                   boundary_gap=first-dk if first is not None and dk is not None else None)
    return dict(states=predicted, mask=mask, selected=selected, dropped=dropped, inside=inside,
                raw_speeds=raw_speeds, metrics=metrics)


def quantiles(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return dict(n=len(a), **({k: float(np.quantile(a, q)) for k, q in
                [('min', 0), ('median', .5), ('p75', .75), ('p90', .9), ('p95', .95), ('max', 1)]} if len(a) else {}))


def balanced_sample(rows, n, seed):
    """Round-robin log strata, randomly permuted within each log; stable input ordering."""
    rng = random.Random(seed)
    groups = {}
    for r in sorted(rows, key=lambda x: x['path']):
        groups.setdefault(r['log_name'], []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups)
    rng.shuffle(keys)
    result = []
    while len(result) < min(n, len(rows)):
        for key in keys:
            if groups[key] and len(result) < n:
                result.append(groups[key].pop())
    return result
