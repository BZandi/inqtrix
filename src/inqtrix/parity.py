"""Parity tooling for auditing Inqtrix against reference research-agent runs.

This module keeps the verification workflow inside the Inqtrix repository.
It can:

- validate the canonical question contract
- execute test runs against the local HTTP test endpoint
- save run artifacts in the Litellm-compatible schema
- compare a run against baseline artifacts deterministically
- optionally generate an LLM-assisted diagnostic report over the full run traces
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINT = "http://127.0.0.1:5100"
DEFAULT_TIMEOUT = 330
TEST_ENDPOINT = "/v1/test/run"
HEALTH_ENDPOINT = "/health"

DEFAULT_QUESTIONS_FILE = PROJECT_ROOT / "tests" / "integration" / "questions.json"
DEFAULT_BASELINES_DIR = PROJECT_ROOT / "tests" / "integration" / "baselines"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "tests" / "integration" / "runs"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "tests" / "integration" / "reports"
DEFAULT_ANALYSIS_TIMEOUT = 120
CLAIM_REPORT_LIMIT = 5
CLAIM_SNAPSHOT_LIMIT = 3

ALGORITHM_CONTEXT = """
Inqtrix ist ein iterativer Research-Agent mit 5 Hauptschritten:
1. classify: Frage einordnen, Suchmodus festlegen, Aspekte ableiten.
2. plan: Queries fuer die naechste Runde generieren.
3. search: Websuche ausfuehren, Ergebnisse zusammenfassen, Claims extrahieren.
4. evaluate: Evidenzqualitaet, Confidence und Stop-Signale bewerten.
5. answer: finale Antwort mit Quellen synthetisieren.

Bewerte bei Vergleichen nicht nur die Endantwort, sondern auch Query-Strategie,
Confidence-Verlauf, Stop-Logik, Quellen-/Claim-Qualitaet und erkennbare
Algorithmus-Regressionen oder Verbesserungen in den Iterations-Logs.
""".strip()


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    """Canonical parity question specification."""

    id: str
    question: str
    category: str
    expected_keywords: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ParityTolerance:
    """Tolerance bands for semantic parity checks."""

    confidence_delta: int = 1
    exact_round_match: bool = True
    query_delta: int = 3
    citation_fraction: float = 0.10
    min_citation_delta: int = 2
    aspect_coverage_delta: float = 0.15
    source_quality_delta: float = 0.15
    claim_quality_delta: float = 0.15
    evidence_score_delta: int = 2
    count_map_fraction: float = 0.25
    min_count_map_delta: int = 1


@dataclass(frozen=True, slots=True)
class Issue:
    """Single contract or parity issue."""

    level: str
    message: str


@dataclass(frozen=True, slots=True)
class MetricCheck:
    """Single structured metric parity check."""

    key: str
    label: str
    level: str
    status: str
    baseline: Any
    current: Any
    delta: int | float | None = None
    allowed_delta: int | float | None = None
    details: str = ""


def load_questions(path: Path | str = DEFAULT_QUESTIONS_FILE) -> list[QuestionSpec]:
    """Load canonical question specifications from a JSON file."""
    question_path = Path(path)
    with question_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_questions = payload.get("questions", [])
    questions: list[QuestionSpec] = []
    for item in raw_questions:
        questions.append(
            QuestionSpec(
                id=str(item.get("id", "")).strip(),
                question=str(item.get("question", "")).strip(),
                category=str(item.get("category", "")).strip(),
                expected_keywords=tuple(
                    str(keyword).strip()
                    for keyword in item.get("expected_keywords", [])
                    if str(keyword).strip()
                ),
                notes=str(item.get("notes", "")).strip(),
            )
        )
    return questions


def validate_question_contract(questions: list[QuestionSpec]) -> list[Issue]:
    """Validate the local canonical question set."""
    issues: list[Issue] = []
    seen_ids: set[str] = set()

    if not questions:
        return [Issue(level="error", message="Keine Fragen definiert.")]

    for question in questions:
        if not question.id:
            issues.append(Issue(level="error", message="Frage ohne ID gefunden."))
            continue
        if question.id in seen_ids:
            issues.append(Issue(level="error", message=f"Doppelte Frage-ID: {question.id}"))
        seen_ids.add(question.id)

        if not question.question:
            issues.append(Issue(level="error", message=f"Leerer Fragetext bei {question.id}."))
        if not question.category:
            issues.append(Issue(level="error", message=f"Leere Kategorie bei {question.id}."))
        if not question.expected_keywords:
            issues.append(
                Issue(
                    level="warning",
                    message=f"Keine expected_keywords fuer {question.id} hinterlegt.",
                )
            )

    return issues


def compare_question_contracts(
    local_questions: list[QuestionSpec],
    reference_questions: list[QuestionSpec],
) -> list[Issue]:
    """Compare local canonical questions against a reference file."""
    issues: list[Issue] = []
    local_by_id = {question.id: question for question in local_questions}
    reference_by_id = {question.id: question for question in reference_questions}

    local_ids = set(local_by_id)
    reference_ids = set(reference_by_id)

    for missing_id in sorted(reference_ids - local_ids):
        issues.append(Issue(level="error", message=f"Fehlende Referenzfrage: {missing_id}"))
    for extra_id in sorted(local_ids - reference_ids):
        issues.append(Issue(level="warning", message=f"Zusaetzliche lokale Frage: {extra_id}"))

    for question_id in sorted(local_ids & reference_ids):
        local = local_by_id[question_id]
        reference = reference_by_id[question_id]

        if local.question != reference.question:
            issues.append(Issue(level="error", message=f"Fragetext drift bei {question_id}."))
        if local.category != reference.category:
            issues.append(Issue(level="error", message=f"Kategorie drift bei {question_id}."))
        if list(local.expected_keywords) != list(reference.expected_keywords):
            issues.append(
                Issue(level="warning", message=f"expected_keywords drift bei {question_id}.")
            )

    return issues


def load_run_file(path: Path | str) -> dict[str, Any]:
    """Load a run artifact from disk."""
    run_path = Path(path)
    with run_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_baseline_dir(path: Path | str) -> dict[str, Any]:
    """Load per-question baseline files into the combined run schema."""
    baseline_dir = Path(path)
    files = sorted(baseline_dir.glob("q*.json"))
    if not files:
        raise FileNotFoundError(f"Keine Baseline-Dateien gefunden in {baseline_dir}")

    results: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not meta:
            meta = payload.get("meta", {})
        results.append(payload.get("result", payload))

    return {"meta": meta, "results": results}


def import_baselines(
    source_dir: Path | str,
    output_dir: Path | str = DEFAULT_BASELINES_DIR,
) -> list[Path]:
    """Copy Litellm-style baseline files into the local Inqtrix repo."""
    source_path = Path(source_dir)
    files = sorted(source_path.glob("q*.json"))
    if not files:
        raise FileNotFoundError(f"Keine Baseline-Dateien gefunden in {source_path}")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for file_path in files:
        target_path = target_dir / file_path.name
        shutil.copy2(file_path, target_path)
        copied.append(target_path)

    migrated_file = source_path / "baseline.json.migrated"
    if migrated_file.exists():
        target_path = target_dir / migrated_file.name
        shutil.copy2(migrated_file, target_path)
        copied.append(target_path)

    return copied


def normalize_test_result(
    question: QuestionSpec,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Normalize a single /v1/test/run result into the reference schema."""
    return {
        "question_id": question.id,
        "question": question.question,
        "category": question.category,
        "answer": result.get("answer", ""),
        "metrics": result.get("metrics", {}),
        "iteration_logs": result.get("iteration_logs", []),
        "top_sources": result.get("top_sources", []),
        "top_claims": result.get("top_claims", []),
    }


def _timestamp_token() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _build_output_path(
    output_dir: Path | str,
    *,
    filename_prefix: str,
    suffix: str,
    stamp: str | None = None,
) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = stamp or _timestamp_token()
    return target_dir / f"{filename_prefix}_{timestamp}{suffix}"


def _write_text_artifact(
    text: str,
    *,
    output_dir: Path | str,
    filename_prefix: str,
    suffix: str,
    stamp: str | None = None,
) -> Path:
    target_path = _build_output_path(
        output_dir,
        filename_prefix=filename_prefix,
        suffix=suffix,
        stamp=stamp,
    )
    with target_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    return target_path


def save_run(
    run_data: dict[str, Any],
    output_dir: Path | str = DEFAULT_RUNS_DIR,
    *,
    stamp: str | None = None,
) -> Path:
    """Save a run artifact with a timestamped filename."""
    run_path = _build_output_path(
        output_dir,
        filename_prefix="run",
        suffix=".json",
        stamp=stamp,
    )
    with run_path.open("w", encoding="utf-8") as handle:
        json.dump(run_data, handle, ensure_ascii=False, indent=2)
    return run_path


def save_report(
    report: dict[str, Any],
    output_dir: Path | str = DEFAULT_REPORTS_DIR,
    *,
    filename_prefix: str = "report",
    stamp: str | None = None,
) -> Path:
    """Save a parity report with a timestamped filename."""
    report_path = _build_output_path(
        output_dir,
        filename_prefix=filename_prefix,
        suffix=".json",
        stamp=stamp,
    )
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report_path


def save_compare_markdown_report(
    report_text: str,
    *,
    output_dir: Path | str = DEFAULT_REPORTS_DIR,
    stamp: str | None = None,
) -> Path:
    """Save the deterministic compare report as Markdown."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = (
        "# Inqtrix Parity Compare Report\n\n"
        f"- erstellt: {created_at}\n\n"
        f"{report_text}\n"
    )
    return _write_text_artifact(
        body,
        output_dir=output_dir,
        filename_prefix="report",
        suffix=".md",
        stamp=stamp,
    )


def save_csv_report(
    csv_text: str,
    *,
    output_dir: Path | str = DEFAULT_REPORTS_DIR,
    filename_prefix: str = "report",
    stamp: str | None = None,
) -> Path:
    """Save a flat CSV export for parity check rows."""
    return _write_text_artifact(
        csv_text,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        suffix=".csv",
        stamp=stamp,
    )


def save_markdown_report(
    report_text: str,
    *,
    analysis_model: str,
    output_dir: Path | str = DEFAULT_REPORTS_DIR,
    filename_prefix: str = "analysis",
    stamp: str | None = None,
) -> Path:
    """Save an LLM-assisted parity report as Markdown."""
    header = (
        "# Inqtrix Parity Analysis\n\n"
        f"- erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- analyse-modell: {analysis_model}\n\n"
        "---\n\n"
    )
    return _write_text_artifact(
        header + report_text,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        suffix=".md",
        stamp=stamp,
    )


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Perform a JSON HTTP request using the standard library only."""
    request_data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        request_data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib_request.Request(url, data=request_data, headers=headers, method=method)
    with urllib_request.urlopen(req, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset))


def run_questions_via_http(
    *,
    endpoint: str,
    questions: list[QuestionSpec],
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Execute canonical questions against the local HTTP test endpoint."""
    health: dict[str, Any] = {}
    try:
        health = _http_json(f"{endpoint}{HEALTH_ENDPOINT}", timeout=15)
    except Exception:
        health = {}

    results: list[dict[str, Any]] = []
    for question in questions:
        raw = _http_json(
            f"{endpoint}{TEST_ENDPOINT}",
            method="POST",
            payload={"question": question.question},
            timeout=timeout,
        )
        results.append(normalize_test_result(question, raw))

    return {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "agent_version": health.get("status", "unknown"),
            "reasoning_model": health.get("reasoning_model", ""),
            "search_model": health.get("search_model", ""),
            "classify_model": health.get("classify_model", ""),
            "summarize_model": health.get("summarize_model", ""),
            "evaluate_model": health.get("evaluate_model", ""),
            "testing_mode": health.get("testing_mode", False),
        },
        "results": results,
    }


def _citation_delta_allowed(reference_value: int, tolerance: ParityTolerance) -> int:
    return max(
        tolerance.min_citation_delta,
        int(math.ceil(reference_value * tolerance.citation_fraction)),
    )


def _count_delta_allowed(reference_value: int, tolerance: ParityTolerance) -> int:
    return max(
        tolerance.min_count_map_delta,
        int(math.ceil(reference_value * tolerance.count_map_fraction)),
    )


def _keyword_hits(answer: str, expected_keywords: tuple[str, ...]) -> int:
    answer_lower = (answer or "").lower()
    return sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)


def _append_issue(issues: list[Issue], level: str, message: str) -> None:
    issues.append(Issue(level=level, message=message))


def _numeric_delta(baseline: Any, current: Any) -> int | float | None:
    if isinstance(baseline, bool) or isinstance(current, bool):
        return None
    if isinstance(baseline, (int, float)) and isinstance(current, (int, float)):
        return current - baseline
    return None


def _check_status(level: str, exceeded: bool) -> str:
    if not exceeded:
        return "pass"
    return "fail" if level == "error" else "warn"


def _append_check(
    checks: list[MetricCheck],
    *,
    key: str,
    label: str,
    level: str,
    exceeded: bool,
    baseline: Any,
    current: Any,
    delta: int | float | None = None,
    allowed_delta: int | float | None = None,
    details: str = "",
) -> None:
    checks.append(
        MetricCheck(
            key=key,
            label=label,
            level=level,
            status=_check_status(level, exceeded),
            baseline=baseline,
            current=current,
            delta=delta,
            allowed_delta=allowed_delta,
            details=details,
        )
    )


def _summarize_checks(checks: list[MetricCheck]) -> dict[str, int]:
    summary = {"total": len(checks), "passed": 0, "warnings": 0, "failed": 0}
    for check in checks:
        if check.status == "fail":
            summary["failed"] += 1
        elif check.status == "warn":
            summary["warnings"] += 1
        else:
            summary["passed"] += 1
    return summary


def compare_runs(
    baseline: dict[str, Any],
    current: dict[str, Any],
    questions: list[QuestionSpec],
    tolerance: ParityTolerance | None = None,
) -> dict[str, Any]:
    """Compare two run artifacts under semantic parity tolerances."""
    tolerance = tolerance or ParityTolerance()
    baseline_by_id = {
        result.get("question_id"): result
        for result in baseline.get("results", [])
        if result.get("question_id")
    }
    current_by_id = {
        result.get("question_id"): result
        for result in current.get("results", [])
        if result.get("question_id")
    }

    question_reports: list[dict[str, Any]] = []
    summary = {
        "total": 0,
        "passed": 0,
        "warnings": 0,
        "failed": 0,
        "checks": {"total": 0, "passed": 0, "warnings": 0, "failed": 0},
    }

    for question in questions:
        summary["total"] += 1
        issues: list[Issue] = []
        checks: list[MetricCheck] = []
        baseline_result = baseline_by_id.get(question.id)
        current_result = current_by_id.get(question.id)

        if baseline_result is None:
            message = f"Baseline fehlt fuer {question.id}."
            _append_check(
                checks,
                key="baseline_presence",
                label="Baseline artifact present",
                level="error",
                exceeded=True,
                baseline=True,
                current=False,
                allowed_delta=0,
                details=message,
            )
            _append_issue(issues, "error", message)
        if current_result is None:
            message = f"Aktueller Run fehlt fuer {question.id}."
            _append_check(
                checks,
                key="current_run_presence",
                label="Current run artifact present",
                level="error",
                exceeded=True,
                baseline=True,
                current=False,
                allowed_delta=0,
                details=message,
            )
            _append_issue(issues, "error", message)

        baseline_metrics = (baseline_result or {}).get("metrics", {})
        current_metrics = (current_result or {}).get("metrics", {})

        if baseline_result is not None and current_result is not None:
            baseline_conf = int(baseline_metrics.get("final_confidence", 0) or 0)
            current_conf = int(current_metrics.get("final_confidence", 0) or 0)
            confidence_delta = current_conf - baseline_conf
            confidence_exceeded = abs(confidence_delta) > tolerance.confidence_delta
            message = f"Confidence drift bei {question.id}: {current_conf} vs {baseline_conf}."
            _append_check(
                checks,
                key="final_confidence",
                label="Confidence",
                level="error",
                exceeded=confidence_exceeded,
                baseline=baseline_conf,
                current=current_conf,
                delta=confidence_delta,
                allowed_delta=tolerance.confidence_delta,
                details=message if confidence_exceeded else "",
            )
            if confidence_exceeded:
                _append_issue(
                    issues,
                    "error",
                    message,
                )

            baseline_rounds = int(baseline_metrics.get("total_rounds", 0) or 0)
            current_rounds = int(current_metrics.get("total_rounds", 0) or 0)
            rounds_delta = current_rounds - baseline_rounds
            rounds_exceeded = tolerance.exact_round_match and baseline_rounds != current_rounds
            message = f"Rundenzahl drift bei {question.id}: {current_rounds} vs {baseline_rounds}."
            _append_check(
                checks,
                key="total_rounds",
                label="Rounds",
                level="error",
                exceeded=rounds_exceeded,
                baseline=baseline_rounds,
                current=current_rounds,
                delta=rounds_delta,
                allowed_delta=0 if tolerance.exact_round_match else None,
                details=message if rounds_exceeded else "",
            )
            if rounds_exceeded:
                _append_issue(
                    issues,
                    "error",
                    message,
                )

            baseline_queries = int(baseline_metrics.get("total_queries", 0) or 0)
            current_queries = int(current_metrics.get("total_queries", 0) or 0)
            query_delta = current_queries - baseline_queries
            query_exceeded = abs(query_delta) > tolerance.query_delta
            message = f"Query-Drift bei {question.id}: {current_queries} vs {baseline_queries}."
            _append_check(
                checks,
                key="total_queries",
                label="Queries",
                level="warning",
                exceeded=query_exceeded,
                baseline=baseline_queries,
                current=current_queries,
                delta=query_delta,
                allowed_delta=tolerance.query_delta,
                details=message if query_exceeded else "",
            )
            if query_exceeded:
                _append_issue(
                    issues,
                    "warning",
                    message,
                )

            baseline_citations = int(baseline_metrics.get("total_citations", 0) or 0)
            current_citations = int(current_metrics.get("total_citations", 0) or 0)
            citation_delta = current_citations - baseline_citations
            allowed_citation_delta = _citation_delta_allowed(baseline_citations, tolerance)
            citation_exceeded = abs(citation_delta) > allowed_citation_delta
            message = (
                f"Zitations-Drift bei {question.id}: {current_citations} vs {baseline_citations}."
            )
            _append_check(
                checks,
                key="total_citations",
                label="Citations",
                level="error",
                exceeded=citation_exceeded,
                baseline=baseline_citations,
                current=current_citations,
                delta=citation_delta,
                allowed_delta=allowed_citation_delta,
                details=message if citation_exceeded else "",
            )
            if citation_exceeded:
                _append_issue(
                    issues,
                    "error",
                    message,
                )

            for key, label, delta in (
                ("aspect_coverage_final", "Aspect coverage", tolerance.aspect_coverage_delta),
                ("source_quality_score_final", "Source quality", tolerance.source_quality_delta),
                ("claim_quality_score_final", "Claim quality", tolerance.claim_quality_delta),
            ):
                baseline_value = float(baseline_metrics.get(key, 0.0) or 0.0)
                current_value = float(current_metrics.get(key, 0.0) or 0.0)
                value_delta = current_value - baseline_value
                exceeded = abs(value_delta) > delta
                message = (
                    f"{label} drift bei {question.id}: "
                    f"{current_value:.3f} vs {baseline_value:.3f}."
                )
                _append_check(
                    checks,
                    key=key,
                    label=label,
                    level="error",
                    exceeded=exceeded,
                    baseline=baseline_value,
                    current=current_value,
                    delta=value_delta,
                    allowed_delta=delta,
                    details=message if exceeded else "",
                )
                if exceeded:
                    _append_issue(
                        issues,
                        "error",
                        message,
                    )

            for key, label in (
                ("evidence_consistency_final", "Evidence consistency"),
                ("evidence_sufficiency_final", "Evidence sufficiency"),
            ):
                baseline_value = int(baseline_metrics.get(key, 0) or 0)
                current_value = int(current_metrics.get(key, 0) or 0)
                value_delta = current_value - baseline_value
                exceeded = abs(value_delta) > tolerance.evidence_score_delta
                message = f"{label} drift bei {question.id}: {current_value} vs {baseline_value}."
                _append_check(
                    checks,
                    key=key,
                    label=label,
                    level="error",
                    exceeded=exceeded,
                    baseline=baseline_value,
                    current=current_value,
                    delta=value_delta,
                    allowed_delta=tolerance.evidence_score_delta,
                    details=message if exceeded else "",
                )
                if exceeded:
                    _append_issue(
                        issues,
                        "error",
                        message,
                    )

            for key in (
                "competing_events_detected",
                "stagnation_detected",
                "falsification_triggered",
                "utility_stop_triggered",
                "plateau_stop_triggered",
            ):
                baseline_value = bool(baseline_metrics.get(key, False))
                current_value = bool(current_metrics.get(key, False))
                exceeded = baseline_value != current_value
                message = f"Stop-/Signal-Drift bei {question.id}: {key} differs."
                _append_check(
                    checks,
                    key=key,
                    label=key,
                    level="error",
                    exceeded=exceeded,
                    baseline=baseline_value,
                    current=current_value,
                    allowed_delta=0,
                    details=message if exceeded else "",
                )
                if exceeded:
                    _append_issue(
                        issues,
                        "error",
                        message,
                    )

            for key, label in (
                ("source_tier_counts_final", "source tier counts"),
                ("claim_status_counts_final", "claim status counts"),
            ):
                baseline_counts = baseline_metrics.get(key, {}) or {}
                current_counts = current_metrics.get(key, {}) or {}
                for count_key in sorted(set(baseline_counts) | set(current_counts)):
                    baseline_value = int(baseline_counts.get(count_key, 0) or 0)
                    current_value = int(current_counts.get(count_key, 0) or 0)
                    allowed_count_delta = _count_delta_allowed(baseline_value, tolerance)
                    count_delta = current_value - baseline_value
                    exceeded = abs(count_delta) > allowed_count_delta
                    message = (
                        f"{label} drift bei {question.id}: {count_key}="
                        f"{current_value} vs {baseline_value}."
                    )
                    _append_check(
                        checks,
                        key=f"{key}.{count_key}",
                        label=f"{label}: {count_key}",
                        level="warning",
                        exceeded=exceeded,
                        baseline=baseline_value,
                        current=current_value,
                        delta=count_delta,
                        allowed_delta=allowed_count_delta,
                        details=message if exceeded else "",
                    )
                    if exceeded:
                        _append_issue(
                            issues,
                            "warning",
                            message,
                        )

            answer = str(current_result.get("answer", ""))
            hit_count = _keyword_hits(answer, question.expected_keywords)
            expected_keyword_count = len(question.expected_keywords)
            missing_keywords = expected_keyword_count > 0 and hit_count == 0
            message = (
                f"Keine expected_keywords im aktuellen Answer fuer {question.id} gefunden."
            )
            _append_check(
                checks,
                key="expected_keywords",
                label="Expected keyword hits",
                level="warning",
                exceeded=missing_keywords,
                baseline=expected_keyword_count,
                current=hit_count,
                delta=hit_count - expected_keyword_count if expected_keyword_count else 0,
                details=message if missing_keywords else "",
            )
            if missing_keywords:
                _append_issue(
                    issues,
                    "warning",
                    message,
                )

        check_summary = _summarize_checks(checks)
        for key, value in check_summary.items():
            summary["checks"][key] += value

        if any(issue.level == "error" for issue in issues):
            status = "fail"
            summary["failed"] += 1
        elif issues:
            status = "warn"
            summary["warnings"] += 1
        else:
            status = "pass"
            summary["passed"] += 1

        question_reports.append(
            {
                "question_id": question.id,
                "question": question.question,
                "category": question.category,
                "status": status,
                "issues": [asdict(issue) for issue in issues],
                "checks": [asdict(check) for check in checks],
                "check_summary": check_summary,
                "baseline_metrics": baseline_metrics,
                "current_metrics": current_metrics,
                "baseline_top_claims": list((baseline_result or {}).get("top_claims", []) or []),
                "current_top_claims": list((current_result or {}).get("top_claims", []) or []),
            }
        )

    return {
        "summary": summary,
        "baseline_meta": baseline.get("meta", {}),
        "current_meta": current.get("meta", {}),
        "questions": question_reports,
    }


def build_contract_report(
    *,
    questions: list[QuestionSpec],
    reference_questions: list[QuestionSpec] | None = None,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a local contract report for canonical parity assets."""
    issues = validate_question_contract(questions)

    if reference_questions is not None:
        issues.extend(compare_question_contracts(questions, reference_questions))

    if baseline is not None:
        baseline_ids = {
            result.get("question_id")
            for result in baseline.get("results", [])
            if result.get("question_id")
        }
        for question in questions:
            if question.id not in baseline_ids:
                issues.append(
                    Issue(
                        level="warning",
                        message=f"Keine Baseline fuer {question.id} im uebergebenen Verzeichnis.",
                    )
                )

    summary = {
        "errors": sum(1 for issue in issues if issue.level == "error"),
        "warnings": sum(1 for issue in issues if issue.level == "warning"),
        "question_count": len(questions),
    }
    return {"summary": summary, "issues": [asdict(issue) for issue in issues]}


def format_contract_report(report: dict[str, Any]) -> str:
    """Render a human-readable contract report."""
    summary = report["summary"]
    lines = [
        "Question contract report",
        f"- questions: {summary['question_count']}",
        f"- errors: {summary['errors']}",
        f"- warnings: {summary['warnings']}",
    ]
    for issue in report.get("issues", []):
        lines.append(f"- {issue['level']}: {issue['message']}")
    return "\n".join(lines)


def build_compare_overview_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-question overview rows for human and file-based reporting."""
    rows: list[dict[str, Any]] = []
    for question in report.get("questions", []):
        baseline_metrics = question.get("baseline_metrics", {}) or {}
        current_metrics = question.get("current_metrics", {}) or {}
        issue_failures = sum(
            1 for issue in question.get("issues", []) if issue.get("level") == "error"
        )
        issue_warnings = sum(
            1 for issue in question.get("issues", []) if issue.get("level") == "warning"
        )
        check_summary = question.get("check_summary", {}) or {}
        rows.append(
            {
                "question_id": question.get("question_id", ""),
                "category": question.get("category", ""),
                "status": question.get("status", ""),
                "failed_checks": max(int(check_summary.get("failed", 0) or 0), issue_failures),
                "warning_checks": max(
                    int(check_summary.get("warnings", 0) or 0),
                    issue_warnings,
                ),
                "baseline_confidence": baseline_metrics.get("final_confidence"),
                "current_confidence": current_metrics.get("final_confidence"),
                "confidence_delta": _numeric_delta(
                    baseline_metrics.get("final_confidence"),
                    current_metrics.get("final_confidence"),
                ),
                "baseline_rounds": baseline_metrics.get("total_rounds"),
                "current_rounds": current_metrics.get("total_rounds"),
                "rounds_delta": _numeric_delta(
                    baseline_metrics.get("total_rounds"),
                    current_metrics.get("total_rounds"),
                ),
                "baseline_queries": baseline_metrics.get("total_queries"),
                "current_queries": current_metrics.get("total_queries"),
                "queries_delta": _numeric_delta(
                    baseline_metrics.get("total_queries"),
                    current_metrics.get("total_queries"),
                ),
                "baseline_citations": baseline_metrics.get("total_citations"),
                "current_citations": current_metrics.get("total_citations"),
                "citations_delta": _numeric_delta(
                    baseline_metrics.get("total_citations"),
                    current_metrics.get("total_citations"),
                ),
                "baseline_aspect_coverage": baseline_metrics.get("aspect_coverage_final"),
                "current_aspect_coverage": current_metrics.get("aspect_coverage_final"),
                "aspect_coverage_delta": _numeric_delta(
                    baseline_metrics.get("aspect_coverage_final"),
                    current_metrics.get("aspect_coverage_final"),
                ),
                "baseline_source_quality": baseline_metrics.get("source_quality_score_final"),
                "current_source_quality": current_metrics.get("source_quality_score_final"),
                "source_quality_delta": _numeric_delta(
                    baseline_metrics.get("source_quality_score_final"),
                    current_metrics.get("source_quality_score_final"),
                ),
                "baseline_claim_quality": baseline_metrics.get("claim_quality_score_final"),
                "current_claim_quality": current_metrics.get("claim_quality_score_final"),
                "claim_quality_delta": _numeric_delta(
                    baseline_metrics.get("claim_quality_score_final"),
                    current_metrics.get("claim_quality_score_final"),
                ),
            }
        )
    return rows


def build_compare_check_rows(
    report: dict[str, Any],
    *,
    only_flagged: bool = False,
) -> list[dict[str, Any]]:
    """Flatten question x metric checks into export rows."""
    rows: list[dict[str, Any]] = []
    for question in report.get("questions", []):
        baseline_top_claims = _serialize_claim_snapshot(
            question.get("baseline_top_claims", []) or []
        )
        current_top_claims = _serialize_claim_snapshot(
            question.get("current_top_claims", []) or []
        )
        for check in question.get("checks", []):
            if only_flagged and check.get("status") == "pass":
                continue
            rows.append(
                {
                    "question_id": question.get("question_id", ""),
                    "question": question.get("question", ""),
                    "category": question.get("category", ""),
                    "question_status": question.get("status", ""),
                    "metric_key": check.get("key", ""),
                    "metric_label": check.get("label", ""),
                    "level": check.get("level", ""),
                    "status": check.get("status", ""),
                    "baseline": check.get("baseline"),
                    "current": check.get("current"),
                    "delta": check.get("delta"),
                    "allowed_delta": check.get("allowed_delta"),
                    "details": check.get("details", ""),
                    "baseline_top_claims": baseline_top_claims,
                    "current_top_claims": current_top_claims,
                }
            )
    return rows


def _claim_tier_summary(claim: dict[str, Any]) -> str:
    counts = claim.get("source_tier_counts", {}) or {}
    parts: list[str] = []
    for key in ("primary", "mainstream", "stakeholder", "unknown", "low"):
        value = int(counts.get(key, 0) or 0)
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "-"


def _claim_table_rows(
    claims: list[dict[str, Any]],
    *,
    max_items: int = CLAIM_REPORT_LIMIT,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, claim in enumerate(claims[:max_items], start=1):
        rows.append(
            {
                "rank": index,
                "status": claim.get("status", ""),
                "claim_type": claim.get("claim_type", ""),
                "needs_primary": "yes" if claim.get("needs_primary", False) else "no",
                "support_count": int(claim.get("support_count", 0) or 0),
                "contradict_count": int(claim.get("contradict_count", 0) or 0),
                "source_tiers": _claim_tier_summary(claim),
                "status_reason": _truncate_text(
                    str(claim.get("status_reason", "") or ""),
                    120,
                ),
                "text": _truncate_text(str(claim.get("text", "") or ""), 180),
            }
        )
    return rows


def _serialize_claim_snapshot(
    claims: list[dict[str, Any]],
    *,
    max_items: int = CLAIM_SNAPSHOT_LIMIT,
    max_chars: int = 600,
) -> str:
    if not claims:
        return ""

    compact: list[dict[str, Any]] = []
    for claim in claims[:max_items]:
        compact.append(
            {
                "text": str(claim.get("text", "") or ""),
                "status": claim.get("status", ""),
                "claim_type": claim.get("claim_type", ""),
                "needs_primary": bool(claim.get("needs_primary", False)),
                "status_reason": str(claim.get("status_reason", "") or ""),
                "support_count": int(claim.get("support_count", 0) or 0),
                "contradict_count": int(claim.get("contradict_count", 0) or 0),
                "source_tier_counts": claim.get("source_tier_counts", {}) or {},
            }
        )

    return _truncate_text(
        json.dumps(compact, ensure_ascii=False),
        max_chars,
    ).replace("\n", " ")


def _format_claim_sections(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for question in report.get("questions", []):
        current_claims = question.get("current_top_claims", []) or []
        baseline_claims = question.get("baseline_top_claims", []) or []
        if not current_claims and not baseline_claims:
            continue

        lines.extend(
            [
                "",
                f"### {question.get('question_id', '')}",
                "",
                "Current top claims:",
                "",
            ]
        )
        if current_claims:
            lines.append(
                _format_markdown_table(
                    _claim_table_rows(current_claims),
                    [
                        ("rank", "#"),
                        ("status", "Status"),
                        ("claim_type", "Type"),
                        ("needs_primary", "Primary?"),
                        ("support_count", "Support"),
                        ("contradict_count", "Contradict"),
                        ("source_tiers", "Source tiers"),
                        ("status_reason", "Reason"),
                        ("text", "Text"),
                    ],
                )
            )
        else:
            lines.append("No exported claims.")

        lines.extend(["", "Baseline top claims:", ""])
        if baseline_claims:
            lines.append(
                _format_markdown_table(
                    _claim_table_rows(baseline_claims),
                    [
                        ("rank", "#"),
                        ("status", "Status"),
                        ("claim_type", "Type"),
                        ("needs_primary", "Primary?"),
                        ("support_count", "Support"),
                        ("contradict_count", "Contradict"),
                        ("source_tiers", "Source tiers"),
                        ("status_reason", "Reason"),
                        ("text", "Text"),
                    ],
                )
            )
        else:
            lines.append("No exported claims.")
    return lines


def _format_scalar(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text or "0"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _format_delta(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "-"
    if isinstance(value, int):
        return f"{value:+d}"
    if isinstance(value, float):
        text = f"{value:+.3f}".rstrip("0").rstrip(".")
        return text or "+0"
    return str(value)


def _pair_summary(current: Any, baseline: Any, delta: Any) -> str:
    current_text = _format_scalar(current)
    baseline_text = _format_scalar(baseline)
    delta_text = _format_delta(delta)
    if delta_text == "-":
        return f"{current_text} vs {baseline_text}"
    return f"{current_text} vs {baseline_text} ({delta_text})"


def _escape_markdown_cell(value: Any) -> str:
    return _format_scalar(value).replace("|", "\\|").replace("\n", "<br>")


def _format_markdown_table(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str]],
) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [header, separator]
    for row in rows:
        body.append(
            "| "
            + " | ".join(_escape_markdown_cell(row.get(key)) for key, _ in columns)
            + " |"
        )
    return "\n".join(body)


def format_compare_report(report: dict[str, Any]) -> str:
    """Render a human-readable parity comparison report."""
    summary = report["summary"]
    overview_rows = build_compare_overview_rows(report)
    flagged_rows = build_compare_check_rows(report, only_flagged=True)
    lines = [
        "Parity comparison report",
        f"- total: {summary['total']}",
        f"- passed: {summary['passed']}",
        f"- warnings: {summary['warnings']}",
        f"- failed: {summary['failed']}",
        f"- metric checks: {summary['checks']['total']}",
        f"- metric checks passed: {summary['checks']['passed']}",
        f"- metric checks warnings: {summary['checks']['warnings']}",
        f"- metric checks failed: {summary['checks']['failed']}",
    ]

    if overview_rows:
        lines.extend(
            [
                "",
                "## Question overview",
                "",
                _format_markdown_table(
                    [
                        {
                            "question_id": row["question_id"],
                            "status": row["status"],
                            "failed_checks": row["failed_checks"],
                            "warning_checks": row["warning_checks"],
                            "confidence": _pair_summary(
                                row["current_confidence"],
                                row["baseline_confidence"],
                                row["confidence_delta"],
                            ),
                            "rounds": _pair_summary(
                                row["current_rounds"],
                                row["baseline_rounds"],
                                row["rounds_delta"],
                            ),
                            "queries": _pair_summary(
                                row["current_queries"],
                                row["baseline_queries"],
                                row["queries_delta"],
                            ),
                            "citations": _pair_summary(
                                row["current_citations"],
                                row["baseline_citations"],
                                row["citations_delta"],
                            ),
                            "aspect": _pair_summary(
                                row["current_aspect_coverage"],
                                row["baseline_aspect_coverage"],
                                row["aspect_coverage_delta"],
                            ),
                            "source": _pair_summary(
                                row["current_source_quality"],
                                row["baseline_source_quality"],
                                row["source_quality_delta"],
                            ),
                            "claim": _pair_summary(
                                row["current_claim_quality"],
                                row["baseline_claim_quality"],
                                row["claim_quality_delta"],
                            ),
                        }
                        for row in overview_rows
                    ],
                    [
                        ("question_id", "Question"),
                        ("status", "Status"),
                        ("failed_checks", "Fail"),
                        ("warning_checks", "Warn"),
                        ("confidence", "Confidence"),
                        ("rounds", "Rounds"),
                        ("queries", "Queries"),
                        ("citations", "Citations"),
                        ("aspect", "Aspect"),
                        ("source", "Source"),
                        ("claim", "Claim"),
                    ],
                ),
            ]
        )

    lines.extend(["", "## Flagged checks", ""])
    if flagged_rows:
        lines.append(
            _format_markdown_table(
                flagged_rows,
                [
                    ("question_id", "Question"),
                    ("metric_label", "Metric"),
                    ("status", "Status"),
                    ("level", "Level"),
                    ("current", "Current"),
                    ("baseline", "Baseline"),
                    ("delta", "Delta"),
                    ("allowed_delta", "Allowed"),
                    ("details", "Details"),
                ],
            )
        )
    else:
        lines.append("No flagged checks.")

    claim_sections = _format_claim_sections(report)
    if claim_sections:
        lines.extend(["", "## Claim snapshots"])
        lines.extend(claim_sections)

    return "\n".join(lines)


def format_compare_csv(report: dict[str, Any]) -> str:
    """Render the full question x metric parity matrix as CSV."""
    rows = build_compare_check_rows(report)
    output = io.StringIO()
    fieldnames = [
        "question_id",
        "question",
        "category",
        "question_status",
        "metric_key",
        "metric_label",
        "level",
        "status",
        "baseline",
        "current",
        "delta",
        "allowed_delta",
        "details",
        "baseline_top_claims",
        "current_top_claims",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def resolve_analysis_model_name(
    configured_model: str,
    default_model: str,
    override_model: str | None = None,
) -> str:
    """Resolve the model name for optional LLM-assisted compare analysis."""
    return override_model or configured_model or default_model


def _extract_confidence_sequence(iteration_logs: list[dict[str, Any]]) -> list[int]:
    return [
        int(entry.get("confidence", 0) or 0)
        for entry in iteration_logs
        if entry.get("node") == "evaluate" and entry.get("confidence") is not None
    ]


def _extract_planned_queries(iteration_logs: list[dict[str, Any]]) -> list[str]:
    queries: list[str] = []
    for entry in iteration_logs:
        if entry.get("node") == "plan":
            for query in entry.get("new_queries", []) or []:
                query_text = str(query).strip()
                if query_text:
                    queries.append(query_text)
    return queries


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n... (gekuerzt)"


def _truncate_logs_for_prompt(logs: list[dict[str, Any]], max_chars: int = 5000) -> str:
    """Compact iteration logs for the LLM prompt without dropping core signals."""
    trimmed: list[dict[str, Any]] = []
    for entry in logs:
        compact = dict(entry)
        if "reasoning" in compact:
            compact["reasoning"] = _truncate_text(str(compact["reasoning"]), 500)
        if "sources_summary" in compact:
            sources_summary: list[dict[str, Any]] = []
            for source in compact["sources_summary"]:
                sources_summary.append(
                    {
                        "query": _truncate_text(str(source.get("query", "")), 160),
                        "summary": _truncate_text(str(source.get("summary", "")), 320),
                        "urls": list(source.get("urls", [])[:3]),
                    }
                )
            compact["sources_summary"] = sources_summary
        trimmed.append(compact)

    serialized = json.dumps(trimmed, ensure_ascii=False, indent=1)
    return _truncate_text(serialized, max_chars)


def build_llm_analysis_prompts(
    baseline: dict[str, Any],
    current: dict[str, Any],
    questions: list[QuestionSpec],
) -> tuple[str, str]:
    """Build the system/user prompts for optional LLM-assisted compare analysis."""
    baseline_results = {
        result.get("question_id"): result
        for result in baseline.get("results", [])
        if result.get("question_id")
    }
    current_results = {
        result.get("question_id"): result
        for result in current.get("results", [])
        if result.get("question_id")
    }

    sections: list[str] = []
    for question in questions:
        baseline_result = baseline_results.get(question.id)
        current_result = current_results.get(question.id)
        if baseline_result is None or current_result is None:
            continue

        baseline_metrics = baseline_result.get("metrics", {})
        current_metrics = current_result.get("metrics", {})
        baseline_logs = baseline_result.get("iteration_logs", []) or []
        current_logs = current_result.get("iteration_logs", []) or []

        section = f"""
### {question.id}: {question.question}
Kategorie: {question.category}
Expected keywords: {', '.join(question.expected_keywords) or 'keine'}

Metriken:
- Confidence: baseline={baseline_metrics.get('final_confidence', 'n/a')} aktuell={current_metrics.get('final_confidence', 'n/a')}
- Runden: baseline={baseline_metrics.get('total_rounds', 'n/a')} aktuell={current_metrics.get('total_rounds', 'n/a')}
- Queries: baseline={baseline_metrics.get('total_queries', 'n/a')} aktuell={current_metrics.get('total_queries', 'n/a')}
- Quellen: baseline={baseline_metrics.get('total_citations', 'n/a')} aktuell={current_metrics.get('total_citations', 'n/a')}
- Aspect coverage: baseline={baseline_metrics.get('aspect_coverage_final', 'n/a')} aktuell={current_metrics.get('aspect_coverage_final', 'n/a')}
- Source quality: baseline={baseline_metrics.get('source_quality_score_final', 'n/a')} aktuell={current_metrics.get('source_quality_score_final', 'n/a')}
- Claim quality: baseline={baseline_metrics.get('claim_quality_score_final', 'n/a')} aktuell={current_metrics.get('claim_quality_score_final', 'n/a')}
- Evidence consistency: baseline={baseline_metrics.get('evidence_consistency_final', 'n/a')} aktuell={current_metrics.get('evidence_consistency_final', 'n/a')}
- Evidence sufficiency: baseline={baseline_metrics.get('evidence_sufficiency_final', 'n/a')} aktuell={current_metrics.get('evidence_sufficiency_final', 'n/a')}

Confidence-Verlauf:
- Baseline: {' -> '.join(str(value) for value in _extract_confidence_sequence(baseline_logs)) or 'n/a'}
- Aktuell: {' -> '.join(str(value) for value in _extract_confidence_sequence(current_logs)) or 'n/a'}

Plan-Queries:
- Baseline: {json.dumps(_extract_planned_queries(baseline_logs), ensure_ascii=False)[:800]}
- Aktuell: {json.dumps(_extract_planned_queries(current_logs), ensure_ascii=False)[:800]}

Antwort Baseline:
{_truncate_text(str(baseline_result.get('answer', '')), 900)}

Antwort Aktuell:
{_truncate_text(str(current_result.get('answer', '')), 900)}

Iteration Logs Baseline:
{_truncate_logs_for_prompt(baseline_logs)}

Iteration Logs Aktuell:
{_truncate_logs_for_prompt(current_logs)}
""".strip()
        sections.append(section)

    system_prompt = (
        "Du bist ein erfahrener Evaluator fuer iterative Research-Agenten. "
        "Analysiere Unterschiede zwischen Baseline und aktuellem Lauf praezise, "
        "mit Fokus auf Root Cause, algorithmische Regressionen/Verbesserungen und "
        "inhaltliche Antwortqualitaet. Schreibe auf Deutsch und in Markdown.\n\n"
        + ALGORITHM_CONTEXT
    )

    user_prompt = (
        "Analysiere den folgenden Parity-Vergleich zwischen einer Baseline und einem "
        "aktuellen Inqtrix-Lauf. Verwende die deterministischen Metriken als harte "
        "Signale, aber bewerte zusaetzlich den inhaltlichen und algorithmischen Verlauf.\n\n"
        "Strukturiere den Report in diese Abschnitte:\n"
        "## 1. Executive Summary\n"
        "Kurze Gesamtbewertung mit Verbesserung/Regression und Risiko-Einschaetzung.\n\n"
        "## 2. Analyse pro Frage\n"
        "Pro Frage: Metrik-Interpretation, Antwortqualitaet, wahrscheinlich betroffener "
        "Algorithmus-Schritt (classify/plan/search/evaluate/answer), Confidence-Verlauf, "
        "und eventuelle Stop-Logik-Auffaelligkeiten.\n\n"
        "## 3. Uebergreifende Muster\n"
        "Welche systematischen Muster siehst du ueber mehrere Fragen hinweg?\n\n"
        "## 4. Empfehlungen\n"
        "Konkrete technische Empfehlungen, priorisiert nach Impact.\n\n"
        "## 5. Gesamtbewertung\n"
        "Sollte dieser Lauf als neue Baseline promoted werden? Begruende klar.\n\n"
        f"Baseline-Zeitstempel: {baseline.get('meta', {}).get('timestamp', 'unbekannt')}\n"
        f"Aktueller Lauf-Zeitstempel: {current.get('meta', {}).get('timestamp', 'unbekannt')}\n\n"
        + "\n\n".join(sections)
    )
    return system_prompt, user_prompt


def generate_llm_analysis_report(
    baseline: dict[str, Any],
    current: dict[str, Any],
    questions: list[QuestionSpec],
    *,
    config_path: Path | str | None = None,
    agent_name: str = "default",
    analysis_model: str | None = None,
    llm_provider: Any | None = None,
    default_model: str | None = None,
    analysis_timeout: int | None = None,
) -> tuple[str, str]:
    """Generate an optional LLM-assisted diagnostic report for parity comparison."""
    config = None
    if llm_provider is None:
        from inqtrix.config import load_config
        from inqtrix.config_bridge import config_to_settings, create_providers_from_config
        from inqtrix.providers import create_providers
        from inqtrix.settings import Settings

        config = load_config(config_path)
        if config.providers and config.models:
            settings = config_to_settings(config, agent_name)
            llm_provider = create_providers_from_config(
                config,
                settings=settings,
                agent_name=agent_name,
            ).llm
            default_model = settings.models.reasoning_model
        else:
            settings = Settings()
            llm_provider = create_providers(settings).llm
            default_model = settings.models.reasoning_model

    configured_model = ""
    configured_timeout = DEFAULT_ANALYSIS_TIMEOUT
    if config is not None:
        configured_model = config.parity.analysis_model
        configured_timeout = config.parity.analysis_timeout

    used_model = resolve_analysis_model_name(
        configured_model, default_model or "", analysis_model)
    timeout = analysis_timeout or configured_timeout
    system_prompt, user_prompt = build_llm_analysis_prompts(baseline, current, questions)
    report = llm_provider.complete(
        user_prompt,
        system=system_prompt,
        model=used_model or None,
        timeout=float(timeout),
    )
    return report, used_model or (default_model or "default")


def _resolve_questions(selected_id: str | None, questions: list[QuestionSpec]) -> list[QuestionSpec]:
    if selected_id is None:
        return questions
    return [question for question in questions if question.id == selected_id]


def _command_contract(args: argparse.Namespace) -> int:
    questions = load_questions(args.questions_file)
    reference_questions = load_questions(
        args.reference_questions) if args.reference_questions else None
    baseline = load_baseline_dir(args.baseline_dir) if args.baseline_dir else None
    report = build_contract_report(
        questions=questions,
        reference_questions=reference_questions,
        baseline=baseline,
    )
    print(format_contract_report(report))
    return 1 if report["summary"]["errors"] else 0


def _command_run(args: argparse.Namespace) -> int:
    questions = _resolve_questions(args.question, load_questions(args.questions_file))
    if not questions:
        print(f"Unknown question id: {args.question}", file=sys.stderr)
        return 2

    try:
        run_data = run_questions_via_http(
            endpoint=args.endpoint,
            questions=questions,
            timeout=args.timeout,
        )
    except urllib_error.HTTPError as exc:
        print(f"HTTP error while running parity suite: {exc}", file=sys.stderr)
        return 2
    except urllib_error.URLError as exc:
        print(f"Endpoint unreachable: {exc}", file=sys.stderr)
        return 2

    run_path = save_run(run_data, args.output_dir)
    print(f"Saved run to {run_path}")
    return 0


def _command_compare(args: argparse.Namespace) -> int:
    questions = load_questions(args.questions_file)
    baseline = load_baseline_dir(args.baseline_dir)
    current = load_run_file(args.run_file)
    report = compare_runs(baseline, current, questions)
    rendered_report = format_compare_report(report)
    rendered_csv = format_compare_csv(report)
    print(rendered_report)
    stamp = _timestamp_token()
    report_path = save_report(report, args.report_dir, stamp=stamp)
    markdown_path = save_compare_markdown_report(
        rendered_report,
        output_dir=args.report_dir,
        stamp=stamp,
    )
    csv_path = save_csv_report(rendered_csv, output_dir=args.report_dir, stamp=stamp)
    print(f"Saved report JSON to {report_path}")
    print(f"Saved report Markdown to {markdown_path}")
    print(f"Saved report CSV to {csv_path}")
    if args.llm_analysis:
        try:
            analysis_report, used_model = generate_llm_analysis_report(
                baseline,
                current,
                questions,
                config_path=args.config,
                agent_name=args.agent_name,
                analysis_model=args.analysis_model,
            )
        except Exception as exc:
            print(f"LLM analysis failed: {exc}", file=sys.stderr)
        else:
            if analysis_report.strip():
                print("\nLLM analysis report\n")
                print(analysis_report)
                markdown_path = save_markdown_report(
                    analysis_report,
                    analysis_model=used_model,
                    output_dir=args.report_dir,
                    stamp=stamp,
                )
                print(f"Saved LLM analysis to {markdown_path}")
    return 1 if report["summary"]["failed"] else 0


def _command_import_baselines(args: argparse.Namespace) -> int:
    try:
        copied = import_baselines(args.source_dir, args.output_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Imported {len(copied)} baseline files to {Path(args.output_dir)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the parity CLI parser."""
    parser = argparse.ArgumentParser(description="Inqtrix parity tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract = subparsers.add_parser("contract", help="Validate parity assets")
    contract.add_argument("--questions-file", default=str(DEFAULT_QUESTIONS_FILE))
    contract.add_argument("--reference-questions")
    contract.add_argument("--baseline-dir")
    contract.set_defaults(func=_command_contract)

    run = subparsers.add_parser("run", help="Run canonical questions via /v1/test/run")
    run.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    run.add_argument("--questions-file", default=str(DEFAULT_QUESTIONS_FILE))
    run.add_argument("--question")
    run.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    run.add_argument("--output-dir", default=str(DEFAULT_RUNS_DIR))
    run.set_defaults(func=_command_run)

    compare = subparsers.add_parser("compare", help="Compare a run against baselines")
    compare.add_argument("run_file")
    compare.add_argument("--baseline-dir", default=str(DEFAULT_BASELINES_DIR))
    compare.add_argument("--questions-file", default=str(DEFAULT_QUESTIONS_FILE))
    compare.add_argument("--report-dir", default=str(DEFAULT_REPORTS_DIR))
    compare.add_argument("--llm-analysis", action="store_true")
    compare.add_argument("--analysis-model")
    compare.add_argument("--config")
    compare.add_argument("--agent-name", default="default")
    compare.set_defaults(func=_command_compare)

    import_baseline_parser = subparsers.add_parser(
        "import-baselines",
        help="Copy Litellm baseline files into tests/integration/baselines",
    )
    import_baseline_parser.add_argument("source_dir")
    import_baseline_parser.add_argument("--output-dir", default=str(DEFAULT_BASELINES_DIR))
    import_baseline_parser.set_defaults(func=_command_import_baselines)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for parity tooling."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
