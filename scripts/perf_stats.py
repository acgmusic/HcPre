#!/usr/bin/env python3
"""Statistics over msprof op_summary Task Duration samples.

Usage:
    python3 perf_stats.py <op_summary.csv> [op_name_filter]

Reads the msprof op_summary csv and prints statistics over the "Task
Duration(us)" column: count, min, max, mean, median, stddev, head average
(first 20% samples), tail average (last 20%), middle average (remaining 60%),
plus the raw sorted sample list. The optional op_name_filter selects which op
name gets the detailed statistics (falls back to the busiest op); the per-op
summary table always lists all op names found in the csv.
"""

import csv
import statistics
import sys


def _col(row, *names):
    for n in names:
        v = row.get(n)
        if v:
            return v
    return ""


def load_samples(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for r in rows:
        name = _col(r, "Op Name", "Name", "Kernel Name") or "?"
        try:
            dur = float(_col(r, "Task Duration(us)", "Task Duration"))
        except ValueError:
            continue
        groups.setdefault(name, []).append(dur)
    return groups


def print_stats(name, durs, verbose=True):
    n = len(durs)
    k = max(1, n // 5)  # 20% bucket: 10 runs at n=50
    head, tail = durs[:k], durs[-k:]
    mid = durs[k:n - k] if n - 2 * k > 0 else list(durs)
    print(f"--- {name} ---")
    print(f"  samples         = {n}")
    print(f"  min             = {min(durs):.3f} us")
    print(f"  max             = {max(durs):.3f} us")
    print(f"  mean            = {statistics.mean(durs):.3f} us")
    print(f"  median (P50)    = {statistics.median(durs):.3f} us")
    if n > 1:
        print(f"  stddev          = {statistics.stdev(durs):.3f} us")
    print(f"  head avg        = {statistics.mean(head):.3f} us  (first {k})")
    print(f"  tail avg        = {statistics.mean(tail):.3f} us  (last {k})")
    print(f"  mid avg         = {statistics.mean(mid):.3f} us  (middle {len(mid)})")
    if verbose:
        print(f"  sorted          = {[round(d, 1) for d in sorted(durs)]}")


def main():
    if len(sys.argv) < 2:
        print("usage: perf_stats.py <op_summary.csv> [op_name_filter]")
        return 1
    groups = load_samples(sys.argv[1])
    if not groups:
        print(f"[perf-stats] no Task Duration rows found in {sys.argv[1]}")
        return 1
    print(f"[perf-stats] {sys.argv[1]}")
    focus = sys.argv[2] if len(sys.argv) > 2 else None
    matched = {k: v for k, v in groups.items() if focus in k} if focus else {}
    if not matched:
        if focus:
            print(f"[perf-stats] filter '{focus}' matched nothing; showing the busiest op instead:")
        busiest = max(groups, key=lambda k: len(groups[k]))
        matched = {busiest: groups[busiest]}
    for name, durs in matched.items():
        print_stats(name, durs)
    print("[perf-stats] per-op summary (count / min / mean us):")
    for name, durs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {name:<40} n={len(durs):<4} min={min(durs):<9.3f} mean={statistics.mean(durs):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
