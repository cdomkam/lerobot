"""Compare two ACT control-loop traces (local MPS vs remote HTTP).

Reads the tick JSONL written by run_local_act.py and run_remote_act.py
(plus the matching `.refills.jsonl` file each runner writes) and prints a
side-by-side summary of loop stability, phase breakdowns, queue dynamics,
and refill timing. The two runners use an identical schema so the only
columns that differ are inside the refill events (`inference_ms` for local
vs `http_total_ms`/`jpeg_*` for remote).

Usage
-----

    python scripts/analyze_act_traces.py \\
        --local /tmp/local-act.jsonl \\
        --remote /tmp/remote-act.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def quantiles(values: list[float], qs: list[float]) -> dict[float, float]:
    if not values:
        return {q: float("nan") for q in qs}
    s = sorted(values)
    out = {}
    for q in qs:
        if q <= 0:
            out[q] = s[0]
        elif q >= 1:
            out[q] = s[-1]
        else:
            idx = q * (len(s) - 1)
            lo = int(math.floor(idx))
            hi = int(math.ceil(idx))
            frac = idx - lo
            out[q] = s[lo] * (1 - frac) + s[hi] * frac
    return out


def summarize_numeric(values: list[float]) -> dict[str, float]:
    vs = [v for v in values if v is not None]
    if not vs:
        return {"count": 0, "mean": float("nan"), "min": float("nan"),
                "max": float("nan"), "p50": float("nan"),
                "p95": float("nan"), "p99": float("nan")}
    q = quantiles(vs, [0.5, 0.95, 0.99])
    return {
        "count": len(vs),
        "mean": statistics.mean(vs),
        "min": min(vs),
        "max": max(vs),
        "p50": q[0.5],
        "p95": q[0.95],
        "p99": q[0.99],
    }


def fmt_stats(s: dict[str, float], unit: str = "ms") -> str:
    if s["count"] == 0:
        return "(no data)"
    return (
        f"mean={s['mean']:.1f}{unit}  p50={s['p50']:.1f}  "
        f"p95={s['p95']:.1f}  p99={s['p99']:.1f}  max={s['max']:.1f}  n={s['count']}"
    )


def analyze_tick_log(ticks: list[dict]) -> dict[str, Any]:
    if not ticks:
        return {"n_ticks": 0}
    target_dt_ms = None
    # Try to infer FPS from the median loop_dt.
    loop_dt = [t["loop_dt_ms"] for t in ticks if t.get("loop_dt_ms") is not None]
    if loop_dt:
        target_dt_ms = statistics.median(loop_dt)

    holding = sum(1 for t in ticks if t.get("was_holding"))
    slow = sum(1 for t in ticks if t.get("slow_tick"))
    clamp = sum(1 for t in ticks if t.get("clamp_engaged"))

    chunk_seams = [t["chunk_seam_delta"] for t in ticks if t.get("chunk_seam_delta") is not None]
    obs_ages = [t["obs_used_age_ms"] for t in ticks if t.get("obs_used_age_ms") is not None]
    queue_depths = [t["queue_depth"] for t in ticks if "queue_depth" in t]
    queue_empty = sum(1 for d in queue_depths if d == 0)

    return {
        "n_ticks": len(ticks),
        "wall_duration_s": ticks[-1]["wall_s"] - ticks[0]["wall_s"] if len(ticks) >= 2 else 0.0,
        "target_dt_ms_est": target_dt_ms,
        "loop_dt_ms": summarize_numeric([t.get("loop_dt_ms") for t in ticks]),
        "tick_total_ms": summarize_numeric([t.get("tick_total_ms") for t in ticks]),
        "obs_read_ms": summarize_numeric([t.get("obs_read_ms") for t in ticks]),
        "policy_select_ms": summarize_numeric([t.get("policy_select_ms") for t in ticks]),
        "motor_write_ms": summarize_numeric([t.get("motor_write_ms") for t in ticks]),
        "max_abs_delta_deg": summarize_numeric([t.get("max_abs_delta_deg") for t in ticks]),
        "chunk_seam_delta_deg": summarize_numeric(chunk_seams),
        "obs_used_age_ms": summarize_numeric(obs_ages),
        "queue_depth": summarize_numeric([float(d) for d in queue_depths]),
        "holding_count": holding,
        "holding_pct": 100.0 * holding / len(ticks),
        "slow_tick_count": slow,
        "slow_tick_pct": 100.0 * slow / len(ticks),
        "clamp_engaged_count": clamp,
        "clamp_engaged_pct": 100.0 * clamp / len(ticks),
        "queue_empty_ticks": queue_empty,
        "n_chunk_seams": len(chunk_seams),
    }


def analyze_refill_log(refills: list[dict]) -> dict[str, Any]:
    if not refills:
        return {"n_refills": 0}
    by_trigger = {"blocking": 0, "prefetch": 0, "error": 0}
    outcomes: dict[str, int] = {}
    for ev in refills:
        by_trigger[ev.get("trigger", "?")] = by_trigger.get(ev.get("trigger", "?"), 0) + 1
        outcomes[ev.get("outcome", "?")] = outcomes.get(ev.get("outcome", "?"), 0) + 1

    queue_at_request = summarize_numeric(
        [float(ev["queue_depth_at_request"]) for ev in refills if "queue_depth_at_request" in ev]
    )
    queue_at_landing = summarize_numeric(
        [float(ev["queue_depth_at_landing"]) for ev in refills if "queue_depth_at_landing" in ev]
    )

    # Remote-only fields.
    http_total = summarize_numeric([ev.get("http_total_ms") for ev in refills])
    server_inf_ms = summarize_numeric(
        [(ev["server_inference_us"] / 1000.0) for ev in refills if ev.get("server_inference_us")]
    )
    network_ms = summarize_numeric([ev.get("network_ms") for ev in refills])
    jpeg_encode = summarize_numeric([ev.get("jpeg_encode_ms") for ev in refills])
    jpeg_bytes = summarize_numeric([ev.get("jpeg_bytes_total") for ev in refills])

    # Local-only fields.
    local_inf = summarize_numeric([ev.get("inference_ms") for ev in refills])

    return {
        "n_refills": len(refills),
        "by_trigger": by_trigger,
        "outcomes": outcomes,
        "queue_depth_at_request": queue_at_request,
        "queue_depth_at_landing": queue_at_landing,
        "http_total_ms": http_total,
        "server_inference_ms": server_inf_ms,
        "network_ms": network_ms,
        "jpeg_encode_ms": jpeg_encode,
        "jpeg_bytes_total": jpeg_bytes,
        "local_inference_ms": local_inf,
    }


def print_report(name: str, tick_summary: dict, refill_summary: dict) -> None:
    print(f"\n=== {name} ===")
    if tick_summary["n_ticks"] == 0:
        print("  (no ticks)")
        return
    td = tick_summary.get("target_dt_ms_est")
    td_str = f"{td:.2f}ms" if td is not None else "n/a"
    print(f"  ticks={tick_summary['n_ticks']}  wall={tick_summary['wall_duration_s']:.1f}s  "
          f"target_dt≈{td_str}")
    print(f"  loop_dt          {fmt_stats(tick_summary['loop_dt_ms'])}")
    print(f"  tick_total       {fmt_stats(tick_summary['tick_total_ms'])}")
    print(f"  obs_read         {fmt_stats(tick_summary['obs_read_ms'])}")
    print(f"  policy_select    {fmt_stats(tick_summary['policy_select_ms'])}")
    print(f"  motor_write      {fmt_stats(tick_summary['motor_write_ms'])}")
    print(f"  max_abs_delta    {fmt_stats(tick_summary['max_abs_delta_deg'], unit='deg')}")
    print(f"  chunk_seam_delta {fmt_stats(tick_summary['chunk_seam_delta_deg'], unit='deg')}"
          f"  (n_seams={tick_summary['n_chunk_seams']})")
    print(f"  obs_used_age     {fmt_stats(tick_summary['obs_used_age_ms'])}"
          f"   <- how stale each applied action's observation is")
    print(f"  queue_depth      {fmt_stats(tick_summary['queue_depth'], unit='')}")
    print(f"  holding          {tick_summary['holding_count']} ticks "
          f"({tick_summary['holding_pct']:.1f}%)")
    print(f"  slow_ticks       {tick_summary['slow_tick_count']} "
          f"({tick_summary['slow_tick_pct']:.1f}%)")
    print(f"  clamp_engaged    {tick_summary['clamp_engaged_count']} "
          f"({tick_summary['clamp_engaged_pct']:.1f}%)")
    print(f"  queue_empty      {tick_summary['queue_empty_ticks']} ticks")

    if refill_summary["n_refills"] == 0:
        print("  (no refills)")
        return
    print(f"\n  --- refills (n={refill_summary['n_refills']}) ---")
    print(f"  triggers   {refill_summary['by_trigger']}")
    print(f"  outcomes   {refill_summary['outcomes']}")
    print(f"  queue@req  {fmt_stats(refill_summary['queue_depth_at_request'], unit='')}")
    print(f"  queue@land {fmt_stats(refill_summary['queue_depth_at_landing'], unit='')}")
    if refill_summary["local_inference_ms"]["count"] > 0:
        print(f"  inference  {fmt_stats(refill_summary['local_inference_ms'])} (local MPS)")
    if refill_summary["http_total_ms"]["count"] > 0:
        print(f"  http_total {fmt_stats(refill_summary['http_total_ms'])}")
        print(f"  server_inf {fmt_stats(refill_summary['server_inference_ms'])}")
        print(f"  network    {fmt_stats(refill_summary['network_ms'])}")
        print(f"  jpeg_enc   {fmt_stats(refill_summary['jpeg_encode_ms'])}")
        print(f"  jpeg_bytes {fmt_stats(refill_summary['jpeg_bytes_total'], unit='B')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=Path, help="tick JSONL written by run_local_act.py")
    ap.add_argument("--remote", type=Path, help="tick JSONL written by run_remote_act.py")
    args = ap.parse_args()

    if not args.local and not args.remote:
        ap.error("provide at least one of --local / --remote")

    def _load_pair(p: Path) -> tuple[list[dict], list[dict]]:
        ticks = load_jsonl(p)
        refills_path = p.with_name(p.name + ".refills") if not p.suffix \
            else p.with_suffix(".refills" + p.suffix)
        refills = load_jsonl(refills_path) if refills_path.exists() else []
        return ticks, refills

    if args.local and args.local.exists():
        ticks, refills = _load_pair(args.local)
        print_report(f"LOCAL ({args.local})", analyze_tick_log(ticks), analyze_refill_log(refills))
    if args.remote and args.remote.exists():
        ticks, refills = _load_pair(args.remote)
        print_report(f"REMOTE ({args.remote})", analyze_tick_log(ticks), analyze_refill_log(refills))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
