#!/usr/bin/env python3
"""Aggregate per-host metrics.json into a single results/SUMMARY.md.

Walks results_collected/<host>/.../metrics.json, classifies each model
by metric keys (detection / classification / ocr), and emits a unified
SUMMARY.md at the project root's results/ directory with:

  - Overall table (one row per (model, host, key-metric)) with threshold + status
  - US baseline vs Nomeroff RU comparison (where both ran)
  - Per-run detailed metrics
  - Caveats from the run's own `note`/`model` field

Supports both layout schemes used in this repo:
  - results_collected/<host>/<model>/metrics.json           (e.g. qudata2/)
  - results_collected/<host>/results/<model>/metrics.json   (e.g. ssh9.qudata.ai/)

Usage from project root:

    python deploy/evaluation/aggregate_summary.py
    python deploy/evaluation/aggregate_summary.py --out results/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COLLECTED = ROOT / "results_collected"
DEFAULT_OUT = ROOT / "results" / "SUMMARY.md"

# ─── Family classification ───────────────────────────────────────────────────

KEY_METRICS = {
    "detection": ["precision", "recall", "f1", "map50"],
    "classification": ["top1_accuracy", "top3_accuracy"],
    "ocr": ["char_accuracy", "full_plate_accuracy"],
}


def infer_family(metrics: dict) -> str:
    keys = set(metrics)
    if {"precision", "recall", "f1"} <= keys:
        return "detection"
    if "top1_accuracy" in keys:
        return "classification"
    if "char_accuracy" in keys or "full_plate_accuracy" in keys:
        return "ocr"
    return "unknown"


# ─── Discovery ───────────────────────────────────────────────────────────────


def parse_path(metrics_path: Path, root: Path) -> tuple[str, str]:
    """Return (host, model) from a metrics.json path under results_collected/."""
    parts = metrics_path.relative_to(root).parts
    host = parts[0]
    # Layouts: <host>/<model>/metrics.json or <host>/results/<model>/metrics.json
    model = parts[2] if len(parts) >= 4 and parts[1] == "results" else parts[1]
    return host, model


def discover(collected: Path) -> list[dict]:
    entries = []
    for mj in sorted(collected.glob("**/metrics.json")):
        with mj.open() as f:
            data = json.load(f)
        if "error" in data:
            continue
        metrics = data.get("metrics") or {k: v for k, v in data.items() if k != "thresholds"}
        host, model = parse_path(mj, collected)
        entries.append(
            {
                "host": host,
                "model": model,
                "family": infer_family(metrics),
                "metrics": metrics,
                "thresholds": data.get("thresholds", {}),
                "skipped": data.get("skipped", 0),
                "note": data.get("note") or data.get("model"),
                "source": str(mj.relative_to(ROOT)),
            }
        )
    return entries


# ─── Rendering ───────────────────────────────────────────────────────────────


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, int):
        return str(v)
    return f"`{v}`"


def render_overall(entries: list[dict]) -> list[str]:
    lines = [
        "## Overall Results",
        "",
        "| Model | Family | Host | Metric | Value | Threshold | Status |",
        "|---|---|---|---|---:|---|---|",
    ]
    for e in entries:
        for key in KEY_METRICS.get(e["family"], []):
            if key not in e["metrics"]:
                continue
            v = e["metrics"][key]
            tinfo = e["thresholds"].get(key, {})
            th = tinfo.get("threshold")
            status = tinfo.get("status", "—")
            th_str = f"≥{th}" if th is not None else "—"
            badge = {"PASS": "✅ PASS", "FAIL": "❌ FAIL"}.get(status, status)
            lines.append(
                f"| `{e['model']}` | {e['family']} | `{e['host']}` "
                f"| {key} | {_fmt(v)} | {th_str} | {badge} |"
            )
    lines.append("")
    return lines


COMPARISONS = [
    ("LPD", "lpdnet", "nomeroff_lpd", ["precision", "recall", "f1", "map50"]),
    ("OCR", "lprnet", "nomeroff_ocr", ["char_accuracy", "full_plate_accuracy"]),
]


def render_us_vs_nomeroff(entries: list[dict]) -> list[str]:
    lines = []
    rows = []
    for pipe, us_name, nm_name, metric_list in COMPARISONS:
        us = next((e for e in entries if e["model"] == us_name), None)
        nm = next((e for e in entries if e["model"] == nm_name), None)
        if not (us and nm):
            continue
        for metric in metric_list:
            if metric not in us["metrics"] or metric not in nm["metrics"]:
                continue
            u_v = us["metrics"][metric]
            n_v = nm["metrics"][metric]
            delta = n_v - u_v
            rows.append(
                f"| {pipe} | {metric} | {u_v:.4f} | **{n_v:.4f}** | {delta:+.4f} |"
            )
    if not rows:
        return lines
    lines.extend(
        [
            "## US baseline vs Nomeroff RU (where both ran)",
            "",
            "| Pipeline | Metric | US baseline | Nomeroff RU | Δ |",
            "|---|---|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    return lines


def render_details(entries: list[dict]) -> list[str]:
    lines = ["## Detailed Results", ""]
    for e in sorted(entries, key=lambda x: (x["host"], x["family"], x["model"])):
        lines.append(f"### `{e['model']}` @ `{e['host']}`  ({e['family']})")
        lines.append("")
        for k, v in e["metrics"].items():
            if isinstance(v, dict):
                inner = ", ".join(f"`{kk}`={_fmt(vv)}" for kk, vv in sorted(v.items()))
                lines.append(f"- **{k}**: {inner}")
            else:
                lines.append(f"- **{k}**: {_fmt(v)}")
        if e["skipped"]:
            lines.append(f"- _skipped_: {e['skipped']}")
        if e["note"] and e["note"] != e["model"]:
            lines.append(f"- _note_: {e['note']}")
        lines.append(f"- _source_: `{e['source']}`")
        lines.append("")
    return lines


def render(entries: list[dict]) -> str:
    hosts = sorted({e["host"] for e in entries})
    families = sorted({e["family"] for e in entries})
    lines = [
        "# CARS Model Evaluation — Aggregated Summary",
        "",
        f"_Generated {date.today().isoformat()} by "
        "`deploy/evaluation/aggregate_summary.py` from `results_collected/`._",
        "",
        f"**Runs aggregated:** {len(entries)} across {len(hosts)} host(s): "
        + ", ".join(f"`{h}`" for h in hosts) + ".",
        f"**Families:** {', '.join(families)}.",
        "",
    ]
    lines += render_overall(entries)
    lines += render_us_vs_nomeroff(entries)
    lines += render_details(entries)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collected", type=Path, default=DEFAULT_COLLECTED,
                   help=f"directory to scan (default: {DEFAULT_COLLECTED.relative_to(ROOT)})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output SUMMARY.md path (default: {DEFAULT_OUT.relative_to(ROOT)})")
    args = p.parse_args(argv)

    entries = discover(args.collected)
    if not entries:
        print(f"No metrics.json found under {args.collected}", flush=True)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(entries))
    print(f"Wrote {args.out.relative_to(ROOT)} — {len(entries)} runs from {args.collected.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
