"""Derive a mock-provider timing profile from exported run traces.

The mock provider replaces real LLM, web-search, and embedding endpoints
during load measurement. Its value depends entirely on reproducing the
*shape* of real provider behaviour: how long each operation blocks and how
many tokens it reports. Both are read here from traces of real runs rather
than guessed, so a load run's wall-clock stays comparable to production.

Input is a directory of trace documents as returned by
``GET /v1/admin/runs/{run_id}/trace/export`` (one JSON file per run, any
file name). Output is a profile document consumed by ``mock_providers.py``.

Operations are keyed ``<mode>.<node>.<channel>`` — the node name comes from
the observation's ancestor span, which is the same grouping the tracing
legend assigns. Latency is stored as p50/p95/max; the mock samples a
lognormal fitted to p50/p95 and clamps it at the observed maximum, so a
load run reproduces the real spread instead of a constant delay.

Two artifacts come out of one pass over the same traces, and the split is
deliberate. The timing profile holds only counts and durations, so it is safe
to keep under version control. The response corpus holds the captured provider
answers verbatim — real questions, real answers, real source material — and
therefore stays untracked operator data that each operator regenerates from
their own runs.

Replaying captured answers rather than synthesising them keeps every response
in the exact wire format the pipeline parses, at a realistic size. Search
results are the one exception: search spans carry no captured payload, so the
mock synthesises them. Extracted claims reference search results by position,
not by URL, so synthesised results still resolve.

Usage::

    python calibrate.py <trace-dir> [--out profiles/<name>.json]
                                    [--corpus corpus/]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Any

# Observation names that are structural rather than a provider call: the run
# root and the HTTP entry span both wrap everything below them.
_STRUCTURAL_PREFIXES = ("POST ", "GET ")
_STRUCTURAL_NAMES = {"inqtrix.run"}

# Observation name -> channel the mock serves it on.
_SPAN_CHANNELS = {
    "web_search": "web_search",
    "embedding": "embedding",
    "embeddings": "embedding",
}


def _is_structural(name: Any) -> bool:
    """Report whether an observation only groups other observations."""
    text = str(name or "")
    return text in _STRUCTURAL_NAMES or text.startswith(_STRUCTURAL_PREFIXES)


def _node_of(observation: dict, by_id: dict[str, dict]) -> str:
    """Return the nearest non-structural ancestor name of *observation*."""
    current: dict | None = observation
    seen: set[str] = set()
    node = "unknown"
    while current is not None and current.get("id") not in seen:
        seen.add(current.get("id"))
        name = current.get("name")
        if name and not _is_structural(name):
            node = str(name)
        current = by_id.get(current.get("parentObservationId"))
    return node


def _system_prompt(payload: Any) -> str:
    """Return the system instruction of a captured or incoming request."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        for message in payload:
            if isinstance(message, dict) and message.get("role") == "system":
                return str(message.get("content") or "")
        if payload and isinstance(payload[0], dict):
            return str(payload[0].get("content") or "")
    return ""


def signature(system_prompt: str, length: int = 80) -> str:
    """Return a stable routing signature for a system instruction.

    Each pipeline step opens its instruction with wording unique to that
    step, which makes the opening the natural operation key. The current
    date is injected into several of them and is therefore removed first,
    so the same step yields the same signature on any day.
    """
    text = " ".join(str(system_prompt or "").split())
    lowered = text.lower()
    if lowered.startswith("heutiges datum:"):
        # Drop the injected date and continue at the instruction itself.
        remainder = text.split(":", 1)[1].lstrip()
        parts = remainder.split(" ", 3)
        text = parts[3] if len(parts) > 3 else remainder
    return text[:length].lower()


def _response_text(output: Any) -> str:
    """Return the assistant text carried by a captured observation output.

    Captures appear either as a chat-message list or as the bare content,
    depending on which provider surface produced them.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        for entry in reversed(output):
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            if isinstance(content, str) and content:
                return content
    if isinstance(output, dict):
        content = output.get("content")
        if isinstance(content, str):
            return content
    return ""


def _quantile(values: list[float], fraction: float) -> float:
    """Return the *fraction* quantile using nearest-rank on sorted values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def collect(
    trace_dir: str,
) -> tuple[dict[str, dict], dict[str, Any], dict[str, list[dict]], dict[str, list[str]]]:
    """Aggregate provider-call samples per operation from every trace file.

    Returns four artifacts: the operation table and a provenance record
    naming how many traces and samples it rests on, so a profile can never
    be mistaken for an authored guess; the captured responses keyed by the
    same operation; and the routing signatures that map an incoming request
    back to its operation.

    Only the first two are safe to keep under version control. Responses are
    verbatim provider answers, and a signature is the opening of a system
    instruction, which for agent work restates the user's own task.
    """
    samples: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
    per_run_calls: dict[str, list[int]] = defaultdict(list)
    models: dict[str, set[str]] = defaultdict(set)
    responses: dict[str, list[dict]] = defaultdict(list)
    signatures: dict[str, set[str]] = defaultdict(set)
    traces_read = 0

    for path in sorted(glob.glob(os.path.join(trace_dir, "*.json"))):
        document = json.load(open(path, encoding="utf-8"))
        payload = document.get("payload") or {}
        observations = payload.get("observations") or []
        if not observations:
            continue
        traces_read += 1
        mode = str((payload.get("metadata") or {}).get("mode") or "")
        if not mode:
            # The trace name carries the mode as ``run:<mode>``.
            mode = str(payload.get("name") or "run:unknown").split(":", 1)[-1]
        by_id = {o["id"]: o for o in observations if o.get("id")}

        counts: dict[str, int] = defaultdict(int)
        for observation in observations:
            kind = observation.get("type")
            name = observation.get("name")
            latency = observation.get("latency")
            if not isinstance(latency, (int, float)) or _is_structural(name):
                continue
            if kind == "GENERATION":
                channel = "llm"
            elif kind == "SPAN":
                channel = _SPAN_CHANNELS.get(str(name), "")
                if not channel:
                    continue
            else:
                continue
            node = _node_of(observation, by_id) if channel == "llm" else str(name)
            key = f"{mode}.{node}.{channel}"
            samples[key].append(
                (
                    float(latency),
                    int(observation.get("promptTokens") or 0),
                    int(observation.get("completionTokens") or 0),
                )
            )
            if observation.get("model"):
                models[key].add(str(observation["model"]))
            counts[key] += 1
            if channel == "llm":
                fingerprint = signature(_system_prompt(observation.get("input")))
                if fingerprint:
                    signatures[key].add(fingerprint)
                text = _response_text(observation.get("output"))
                if text:
                    responses[key].append(
                        {
                            "content": text,
                            "prompt_tokens": int(
                                observation.get("promptTokens") or 0
                            ),
                            "completion_tokens": int(
                                observation.get("completionTokens") or 0
                            ),
                        }
                    )
        for key, count in counts.items():
            per_run_calls[key].append(count)

    operations: dict[str, dict] = {}
    total_samples = 0
    for key, rows in sorted(samples.items()):
        latencies = [r[0] for r in rows]
        prompt_tokens = [r[1] for r in rows]
        completion_tokens = [r[2] for r in rows]
        calls = per_run_calls.get(key) or [0]
        total_samples += len(rows)
        operations[key] = {
            "samples": len(rows),
            "latency_seconds": {
                "p50": round(_quantile(latencies, 0.50), 3),
                "p95": round(_quantile(latencies, 0.95), 3),
                "max": round(max(latencies), 3),
            },
            "prompt_tokens_median": int(statistics.median(prompt_tokens)),
            "completion_tokens_median": int(statistics.median(completion_tokens)),
            "calls_per_run_median": round(statistics.median(calls), 1),
            "models": sorted(models.get(key, set())),
        }

    provenance = {
        "traces_read": traces_read,
        "operations": len(operations),
        "provider_call_samples": total_samples,
        "derived_from": (
            "run trace export (GET /v1/admin/runs/{run_id}/trace/export)"
        ),
    }
    routing = {key: sorted(value) for key, value in sorted(signatures.items())}
    return operations, provenance, responses, routing


def main() -> int:
    """Read traces, print a readable table, and write the profile document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", help="Directory of exported trace JSON files")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "profiles", "calibrated.json"),
        help="Profile document to write",
    )
    parser.add_argument(
        "--corpus",
        default=os.path.join(os.path.dirname(__file__), "corpus"),
        help=(
            "Directory for the captured response corpus. Holds real provider "
            "answers and stays untracked."
        ),
    )
    args = parser.parse_args()

    operations, provenance, responses, routing = collect(args.trace_dir)
    if not operations:
        print(f"No provider calls found in {args.trace_dir!r}.")
        return 1

    width = max(len(k) for k in operations)
    print(f"{'operation':{width}s} {'n':>4s} {'p50':>8s} {'p95':>8s} {'max':>8s} {'calls':>6s}")
    print("-" * (width + 38))
    for key, entry in operations.items():
        latency = entry["latency_seconds"]
        print(
            f"{key:{width}s} {entry['samples']:4d} {latency['p50']:8.2f} "
            f"{latency['p95']:8.2f} {latency['max']:8.2f} "
            f"{entry['calls_per_run_median']:6.1f}"
        )

    document = {"provenance": provenance, "operations": operations}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"\n{provenance['provider_call_samples']} provider calls from "
        f"{provenance['traces_read']} traces -> {args.out}"
    )

    os.makedirs(args.corpus, exist_ok=True)
    manifest: dict[str, int] = {}
    corpus_bytes = 0
    for key, entries in sorted(responses.items()):
        path = os.path.join(args.corpus, f"{key}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(entries, handle, ensure_ascii=False)
        manifest[key] = len(entries)
        corpus_bytes += os.path.getsize(path)
    with open(
        os.path.join(args.corpus, "manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "provenance": provenance,
                "responses_per_operation": manifest,
                "routing_signatures": routing,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    missing = sorted(k for k in operations if k.endswith(".llm") and k not in manifest)
    if missing:
        # Silence here would look like a complete corpus at replay time.
        print(f"WARNING: no captured responses for {len(missing)} operation(s):")
        for key in missing:
            print(f"  {key}")
    print(
        f"{sum(manifest.values())} captured responses "
        f"({corpus_bytes / 1024:.0f} KiB) -> {args.corpus}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
