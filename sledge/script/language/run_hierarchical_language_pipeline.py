#!/usr/bin/env python3
"""Run the recursive hierarchical natural-language pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from sledge.semantic_control.language.hierarchical_pipeline import (
    HierarchicalEventFramePipeline,
)


def _records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} must contain one JSON object per line")
            yield item


def _prompt(record: Dict[str, Any]) -> str:
    for key in ("prompt", "text", "input", "description", "natural_language"):
        if key in record:
            return str(record[key])
    raise KeyError(f"record has no prompt-like field: {record}")


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="single natural-language traffic-scene prompt")
    source.add_argument("--input-jsonl", type=Path, help="JSONL file with a prompt-like field")
    parser.add_argument("--output", type=Path, help="optional JSON/JSONL output path")
    parser.add_argument("--llm-provider", choices=["none", "fallback", "ollama"], default="none")
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--no-repair", action="store_true")
    args = parser.parse_args()

    pipeline = HierarchicalEventFramePipeline(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        ollama_url=args.ollama_url,
        no_repair=args.no_repair,
    )

    if args.prompt is not None:
        result = pipeline.parse_to_result(args.prompt)
        payload = {
            "prompt": args.prompt,
            "valid": result.valid,
            "frame_issues": result.frame_issues,
            "spec_issues": result.spec_issues,
            "hierarchy_issues": result.hierarchy_issues,
            "frame": result.frame.to_dict(),
            "spec": result.spec,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return

    assert args.input_jsonl is not None
    output_lines = []
    for record in _records(args.input_jsonl):
        prompt = _prompt(record)
        result = pipeline.parse_to_result(prompt)
        output_lines.append(
            json.dumps(
                {
                    **record,
                    "hierarchical_result": {
                        "valid": result.valid,
                        "frame_issues": result.frame_issues,
                        "spec_issues": result.spec_issues,
                        "hierarchy_issues": result.hierarchy_issues,
                        "frame": result.frame.to_dict(),
                        "spec": result.spec,
                    },
                },
                ensure_ascii=False,
            )
        )

    text = "\n".join(output_lines) + ("\n" if output_lines else "")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
