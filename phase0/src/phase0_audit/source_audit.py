from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sledge_contract import verified_contract, verify_repository_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase-0 assumptions against the checked-out SLEDGE code")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    report = verify_repository_contract()
    payload = {
        "verified_contract": verified_contract(),
        "runtime_verification": report.to_dict(),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if not report.ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
