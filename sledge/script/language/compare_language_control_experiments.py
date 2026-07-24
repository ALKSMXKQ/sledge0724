#!/usr/bin/env python3
"""Compare two natural-language scene-understanding experiment paths.

Both methods are evaluated on the same JSONL cases:

1. eventframe_main:
   text -> EventFrame -> ordered event sequence -> verification/repair
   -> hazard parameter template -> missing-info completion.

2. direct_template_baseline:
   text -> direct semantic slots -> final parameter-template projection.

Gold slots can be stored under any of {"expect", "expected", "expected_slots"}.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sledge.semantic_control.language.direct_template_baseline import (
    DirectTemplateBaseline,
    flatten_dict as flatten_direct_dict,
    validate_direct_template_spec,
)
from sledge.semantic_control.language.event_frame_mapper import (
    EventFrameToHazardSpecMapper,
    flatten_dict as flatten_eventframe_dict,
    validate_spec,
)
from sledge.semantic_control.language.event_frame_parser import EventFrameParser
from sledge.semantic_control.language.event_frame_verifier import EventFrameVerifier
from sledge.semantic_control.language.event_sequence_builder import EventSequenceBuilder
from sledge.semantic_control.language.missing_info_filler import MissingInfoFiller


SCHEMA_ALIASES: Dict[str, set[str]] = {
    "eventframe_hazard_parameter_spec_v2": {
        "direct_template_baseline_spec",
    },
    "direct_template_baseline_spec": {
        "eventframe_hazard_parameter_spec_v2",
    },
}

VALUE_ALIASES: Dict[str, set[str]] = {
    "lateral": {"lateral_crossing"},
    "lateral_crossing": {"lateral"},
    "merging": {"lane_change"},
    "lane_change": {"merging"},
    "longitudinal": {"longitudinal_braking"},
    "longitudinal_braking": {"longitudinal"},
    "crossing_path_conflict": {"lateral_conflict"},
    "lateral_conflict": {"crossing_path_conflict"},
}


@dataclass
class MethodOutput:
    method: str
    parse_success: bool
    check_pass: bool
    compile_success: bool
    frame: Dict[str, Any]
    predicted_spec: Dict[str, Any]
    predicted_flat: Dict[str, Any]
    frame_issues: List[str]
    spec_errors: List[str]
    spec_issues: List[str]
    error_stage: Optional[str] = None
    error: Optional[str] = None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return cases


def prompt_text(case: Dict[str, Any]) -> str:
    for key in ["prompt", "text", "input", "description", "natural_language"]:
        if key in case:
            return str(case[key])
    raise KeyError(f"case has no prompt-like field: {case}")


def gold_slots(case: Dict[str, Any], *, score_schema_version: bool) -> Dict[str, Any]:
    gold = case.get("expected")
    if gold is None:
        gold = case.get("expect")
    if gold is None:
        gold = case.get("expected_slots")
    out = dict(gold or {})
    if not score_schema_version:
        out.pop("schema_version", None)
    return out


def slot_match(predicted: Any, expected: Any) -> bool:
    if predicted == expected:
        return True

    if isinstance(expected, dict) and "any_of" in expected:
        return any(slot_match(predicted, item) for item in expected["any_of"])

    if isinstance(expected, str) and not isinstance(predicted, (dict, list)):
        pred_text = str(predicted)
        if pred_text in SCHEMA_ALIASES.get(expected, set()):
            return True
        if pred_text in VALUE_ALIASES.get(expected, set()):
            return True

        stripped = expected.strip()
        if stripped.startswith("{") and "any_of" in stripped:
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and "any_of" in parsed:
                return any(slot_match(predicted, item) for item in parsed["any_of"])

    if isinstance(predicted, list):
        if isinstance(expected, list):
            return any(item in expected for item in predicted)
        if isinstance(expected, dict) and "any_of" in expected:
            return any(slot_match(item, exp) for item in predicted for exp in expected["any_of"])

    return False


def compare_slots(pred_flat: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[int, int, List[Dict[str, Any]]]:
    correct = 0
    total = 0
    mismatches: List[Dict[str, Any]] = []
    for key, exp in expected.items():
        total += 1
        pred = pred_flat.get(key, "__MISSING__")
        if slot_match(pred, exp):
            correct += 1
        else:
            mismatches.append({"slot": key, "expected": exp, "predicted": pred})
    return correct, total, mismatches


class EventFrameMainExperiment:
    method = "eventframe_main"

    def __init__(
        self,
        *,
        llm_provider: str,
        llm_model: str,
        ollama_url: str,
        no_repair: bool,
        respect_llm_event_sequence: bool,
    ) -> None:
        self.parser = EventFrameParser(
            llm_provider="ollama" if llm_provider == "ollama" else "none",
            llm_model=llm_model,
            ollama_url=ollama_url,
            allow_fallback=True,
        )
        self.sequence_builder = EventSequenceBuilder()
        self.mapper = EventFrameToHazardSpecMapper()
        self.verifier = EventFrameVerifier()
        self.filler = MissingInfoFiller()
        self.no_repair = no_repair
        self.respect_llm_event_sequence = respect_llm_event_sequence

    def run(self, prompt: str) -> MethodOutput:
        stage = "parse"
        try:
            frame = self.parser.parse(prompt)
            stage = "event_sequence"
            frame = self.sequence_builder.build(frame, overwrite=not self.respect_llm_event_sequence)
            stage = "frame_verify"
            verify0 = self.verifier.verify_frame(frame)
            if not verify0.passed and not self.no_repair:
                stage = "frame_repair"
                frame = self.verifier.repair_frame(frame)
                stage = "event_sequence_after_repair"
                frame = self.sequence_builder.build(frame, overwrite=not self.respect_llm_event_sequence)
            stage = "frame_verify_after_repair"
            verify1 = self.verifier.verify_frame(frame)
            stage = "map"
            spec = self.mapper.map(frame)
            stage = "missing_info_fill"
            spec = self.filler.fill(spec, frame)
            stage = "spec_validate"
            ok, spec_errors = validate_spec(spec)
            stage = "spec_verify"
            spec_verify = self.verifier.verify_spec(spec)
            return MethodOutput(
                method=self.method,
                parse_success=True,
                check_pass=verify1.passed,
                compile_success=ok and spec_verify.passed,
                frame=frame.to_dict(),
                predicted_spec=spec,
                predicted_flat=flatten_eventframe_dict(spec),
                frame_issues=verify1.issues,
                spec_errors=spec_errors,
                spec_issues=spec_verify.issues,
            )
        except Exception as exc:
            return MethodOutput(
                method=self.method,
                parse_success=False,
                check_pass=False,
                compile_success=False,
                frame={},
                predicted_spec={},
                predicted_flat={},
                frame_issues=[],
                spec_errors=[],
                spec_issues=[],
                error_stage=stage,
                error=f"{type(exc).__name__}: {exc}",
            )


class DirectTemplateBaselineExperiment:
    method = "direct_template_baseline"

    def __init__(
        self,
        *,
        llm_provider: str,
        llm_model: str,
        ollama_url: str,
    ) -> None:
        self.pipeline = DirectTemplateBaseline(
            llm_provider="ollama" if llm_provider == "ollama" else "none",
            llm_model=llm_model,
            ollama_url=ollama_url,
            allow_fallback=True,
        )

    def run(self, prompt: str) -> MethodOutput:
        stage = "parse"
        try:
            frame, spec = self.pipeline.parse_to_spec(prompt)
            frame_issues: List[str] = []
            slots = frame.semantic_slots
            if slots.get("actor_type") in {"", "unknown"}:
                frame_issues.append("missing_actor_type")
            if slots.get("event_type") in {"", "unknown"}:
                frame_issues.append("missing_event_type")
            if slots.get("conflict_geometry") in {"", "unknown"}:
                frame_issues.append("missing_conflict_geometry")
            if not frame.event_sequence:
                frame_issues.append("missing_event_sequence")
            stage = "spec_validate"
            ok, spec_errors = validate_direct_template_spec(spec)
            return MethodOutput(
                method=self.method,
                parse_success=True,
                check_pass=not frame_issues,
                compile_success=ok,
                frame=frame.to_dict(),
                predicted_spec=spec,
                predicted_flat=flatten_direct_dict(spec),
                frame_issues=frame_issues,
                spec_errors=spec_errors,
                spec_issues=[],
            )
        except Exception as exc:
            return MethodOutput(
                method=self.method,
                parse_success=False,
                check_pass=False,
                compile_success=False,
                frame={},
                predicted_spec={},
                predicted_flat={},
                frame_issues=[],
                spec_errors=[],
                spec_issues=[],
                error_stage=stage,
                error=f"{type(exc).__name__}: {exc}",
            )


def init_summary(methods: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    return {
        method: {
            "num_cases": 0,
            "parse_success": 0,
            "check_pass": 0,
            "compile_success": 0,
            "slot_correct": 0,
            "slot_total": 0,
        }
        for method in methods
    }


def finalize_summary(raw: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for method, stats in raw.items():
        n = stats["num_cases"]
        slot_total = stats["slot_total"]
        summary[method] = {
            "num_cases": n,
            "parse_success_rate": stats["parse_success"] / n if n else 0.0,
            "check_pass_rate": stats["check_pass"] / n if n else 0.0,
            "compile_success_rate": stats["compile_success"] / n if n else 0.0,
            "slot_accuracy": stats["slot_correct"] / slot_total if slot_total else None,
            "slot_correct": stats["slot_correct"],
            "slot_total": slot_total,
        }
    return summary


def write_report(path: Path, summary: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    lines: List[str] = [
        "# Language Control Experiment Comparison",
        "",
        "## Summary",
        "",
        "| Method | Cases | Parse | Check | Compile | Slot accuracy | Slots |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, stats in summary["methods"].items():
        slot_acc = stats["slot_accuracy"]
        slot_acc_text = "n/a" if slot_acc is None else f"{slot_acc:.4f}"
        lines.append(
            "| "
            f"{method} | {stats['num_cases']} | {stats['parse_success_rate']:.4f} | "
            f"{stats['check_pass_rate']:.4f} | {stats['compile_success_rate']:.4f} | "
            f"{slot_acc_text} | {stats['slot_correct']}/{stats['slot_total']} |"
        )

    lines.extend(["", "## Mismatched Cases", ""])
    any_failed = False
    for record in records:
        failed_methods = [
            method
            for method, result in record["results"].items()
            if (
                result.get("error")
                or not result.get("check_pass", True)
                or not result.get("compile_success", True)
                or result.get("mismatches")
                or result.get("spec_validation_errors")
                or result.get("spec_verification_issues")
            )
        ]
        if not failed_methods:
            continue
        any_failed = True
        lines.extend([f"### {record['id']}", "", f"Prompt: `{record['prompt']}`", ""])
        for method in failed_methods:
            result = record["results"][method]
            lines.append(f"#### {method}")
            if result.get("error"):
                lines.append(f"- Error stage: `{result.get('error_stage')}`")
                lines.append(f"- Error: `{result.get('error')}`")
            for mismatch in result.get("mismatches", []):
                lines.append(
                    f"- `{mismatch['slot']}`: expected `{mismatch['expected']}`, predicted `{mismatch['predicted']}`"
                )
            if result.get("frame_verification_issues"):
                lines.append(f"- Frame issues: `{result['frame_verification_issues']}`")
            if result.get("spec_validation_errors"):
                lines.append(f"- Spec validation errors: `{result['spec_validation_errors']}`")
            if result.get("spec_verification_issues"):
                lines.append(f"- Spec verification issues: `{result['spec_verification_issues']}`")
            lines.append("")
    if not any_failed:
        lines.append("No slot mismatches or validation failures.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases_jsonl", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--llm_provider", default="none", choices=["none", "fallback", "ollama"])
    ap.add_argument("--llm_model", default="qwen2.5:7b")
    ap.add_argument("--ollama_url", default="http://127.0.0.1:11434")
    ap.add_argument("--no_repair", action="store_true")
    ap.add_argument("--respect_llm_event_sequence", action="store_true")
    ap.add_argument("--score_schema_version", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_jsonl(args.cases_jsonl)

    experiments = [
        EventFrameMainExperiment(
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            ollama_url=args.ollama_url,
            no_repair=args.no_repair,
            respect_llm_event_sequence=args.respect_llm_event_sequence,
        ),
        DirectTemplateBaselineExperiment(
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            ollama_url=args.ollama_url,
        ),
    ]

    raw_summary = init_summary(exp.method for exp in experiments)
    records: List[Dict[str, Any]] = []
    groups = set()
    categories = set()

    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("id", f"case_{index:04d}"))
        prompt = prompt_text(case)
        expected = gold_slots(case, score_schema_version=args.score_schema_version)
        groups.add(str(case.get("group", "unknown")))
        categories.add(str(case.get("category", "unknown")))
        print(f"[COMPARE] {index}/{len(cases)} {case_id}: {prompt}")

        record: Dict[str, Any] = {
            "id": case_id,
            "category": case.get("category"),
            "group": case.get("group"),
            "prompt": prompt,
            "expected": expected,
            "results": {},
        }

        for experiment in experiments:
            output = experiment.run(prompt)
            correct, total, mismatches = compare_slots(output.predicted_flat, expected)
            stats = raw_summary[experiment.method]
            stats["num_cases"] += 1
            stats["parse_success"] += int(output.parse_success)
            stats["check_pass"] += int(output.check_pass)
            stats["compile_success"] += int(output.compile_success)
            stats["slot_correct"] += correct
            stats["slot_total"] += total
            record["results"][experiment.method] = {
                "parse_success": output.parse_success,
                "check_pass": output.check_pass,
                "compile_success": output.compile_success,
                "error_stage": output.error_stage,
                "error": output.error,
                "frame": output.frame,
                "frame_verification_issues": output.frame_issues,
                "predicted_spec": output.predicted_spec,
                "predicted_flat": output.predicted_flat,
                "spec_validation_errors": output.spec_errors,
                "spec_verification_issues": output.spec_issues,
                "slot_correct": correct,
                "slot_total": total,
                "slot_accuracy": correct / total if total else None,
                "mismatches": mismatches,
            }
        records.append(record)

    summary = {
        "cases_jsonl": str(args.cases_jsonl),
        "num_cases": len(cases),
        "num_categories": len(categories),
        "num_groups": len(groups),
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "score_schema_version": args.score_schema_version,
        "methods": finalize_summary(raw_summary),
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "per_case_predictions.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_report(args.output_dir / "comparison_report.md", summary, records)

    print("\n[DONE] Language control experiment comparison finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

