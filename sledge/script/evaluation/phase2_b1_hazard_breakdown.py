from __future__ import annotations

"""Break down Phase-2 B1 audit results without changing the audit protocol."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


PREDICATES = ("O", "E", "C", "T", "H")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_group(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    built = sum(row.get("builder_status") == "built" for row in rows)
    return {
        "N": n,
        "builder_yield": built / n if n else 0.0,
        **{
            key: sum(bool(row.get(key)) for row in rows) / n if n else 0.0
            for key in PREDICATES
        },
    }


def grouped_summary(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return {
        name: summarize_group(items)
        for name, items in sorted(groups.items(), key=lambda item: item[0])
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-2 subgroup breakdown")
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["condition_id"],
        help="Audit-row keys, e.g. condition_id source_scenario_type",
    )
    args = parser.parse_args()

    rows = _read_jsonl(args.audit_jsonl)
    payload = {
        "schema_version": "phase2_b1_hazard_breakdown_v1",
        "overall": summarize_group(rows),
        "groups": {key: grouped_summary(rows, key) for key in args.group_by},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
