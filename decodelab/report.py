"""Terminal, JSON and Markdown reports for decoding comparisons."""

from __future__ import annotations

import json

METRIC_KEYS = ["tokens", "distinct_1", "distinct_2", "repetition_rate", "entropy_bits"]


def build_report(results: dict[str, dict[str, float]]) -> dict:
    """results maps strategy name to its metric dict."""
    return {
        "strategies": {
            name: {key: round(metrics[key], 4) for key in METRIC_KEYS}
            for name, metrics in results.items()
        }
    }


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def to_markdown(report: dict) -> str:
    lines = [
        "# DecodeLab report",
        "",
        "| Strategy | Tokens | Distinct-1 | Distinct-2 | Repetition | Entropy (bits) |",
        "|---|---|---|---|---|---|",
    ]
    for name, metrics in report["strategies"].items():
        lines.append(
            f"| {name} | {metrics['tokens']:.0f} | {metrics['distinct_1']:.4f} "
            f"| {metrics['distinct_2']:.4f} | {metrics['repetition_rate']:.4f} "
            f"| {metrics['entropy_bits']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def to_terminal(report: dict) -> str:
    lines = [f"{'strategy':<22} {'tokens':>6} {'dist-1':>7} {'dist-2':>7} {'repeat':>7} {'entropy':>8}"]
    for name, metrics in report["strategies"].items():
        lines.append(
            f"{name:<22} {metrics['tokens']:>6.0f} {metrics['distinct_1']:>7.4f} "
            f"{metrics['distinct_2']:>7.4f} {metrics['repetition_rate']:>7.4f} {metrics['entropy_bits']:>8.4f}"
        )
    return "\n".join(lines)
