from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..adapters.explicit_matrix import ExplicitActorMatrixSchema, ExplicitMatrixSceneAdapter
from ..audit.roundtrip import roundtrip_check
from ..core.types import ActorType
from ..core.validation import validate_scene


def _load_schema(path: Path) -> tuple[ExplicitActorMatrixSchema, dict[int, ActorType], str]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    columns = cfg["columns"]
    schema = ExplicitActorMatrixSchema(**columns)
    actor_type_map = {int(k): ActorType(v) for k, v in cfg["actor_type_map"].items()}
    frame = cfg.get("frame", "unspecified")
    return schema, actor_type_map, frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-0 explicit representation audit")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--actors-key", default="actors")
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--scene-id", default="audit_scene")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=False) as data:
        actors = np.asarray(data[args.actors_key])

    schema, type_map, frame = _load_schema(args.schema)
    adapter = ExplicitMatrixSceneAdapter(schema, type_map, frame=frame)
    before, legacy_after, after, report = roundtrip_check(
        adapter, actors, scene_id=args.scene_id
    )

    payload = {
        "scene_id": args.scene_id,
        "legacy_shape": list(actors.shape),
        "mapped_columns": list(schema.mapped_columns),
        "canonical_actor_count": len(before.actors),
        "canonical_validation_errors": validate_scene(before),
        "roundtrip": report.to_dict(),
        "legacy_mapped_max_abs_error": _mapped_error(actors, legacy_after, schema.mapped_columns),
        "actor_type_counts": _actor_type_counts(before),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not report.ok:
        raise SystemExit(2)


def _mapped_error(a: np.ndarray, b: np.ndarray, columns: tuple[int, ...]) -> float:
    if not columns:
        return 0.0
    return float(np.max(np.abs(a[:, columns].astype(float) - b[:, columns].astype(float))))


def _actor_type_counts(scene) -> dict[str, int]:
    result: dict[str, int] = {}
    for actor in scene.actors:
        result[actor.actor_type.value] = result.get(actor.actor_type.value, 0) + 1
    return result


if __name__ == "__main__":
    main()
