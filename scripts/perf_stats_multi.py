#!/usr/bin/env python3
"""Multi-shape msprof op_summary aggregation for hc_pre.

Usage:
    python3 perf_stats_multi.py <op_summary.csv> <shapes.csv> <outdir> [op_name_filter]

Groups the "Task Duration(us)" samples of one msprof run (launched by
hcpre_sim_app.py in HC_PRE_SHAPES mode, round-major: outer=iters, inner=shapes)
back to each shape, then:
  * prints, per shape: the n samples on ONE comma-separated line + the same
    statistics block as perf_stats.py
  * writes <outdir>/perf_multi_summary.txt   (full human-readable summary)
  * writes <outdir>/perf_multi_all.csv       (one line per shape, '#' comments;
                                              compatible with perf_extreme_eval)
  * writes <outdir>/perf_shape/b{b}_bs{bs}_d{d}.csv  (ONE line = one shape's
    samples; THE recommended per-shape input for perf_extreme_eval.py, so the
    extreme-value analysis never pools different shapes)

Grouping strategy: primary = match the op_summary "Input Shape" column against
the expected x shape (both 4D "b*bs*4*d" and folded "bs*4*d" forms); fallback =
positional round-major mapping (launch order), validated by sample count.
"""

import csv
import os
import re
import statistics
import sys

SHAPE_COLS = ("Input Shape", "Input Shapes", "Shapes", "InputShapes")


def _col(row, *names):
    for n in names:
        v = row.get(n)
        if v:
            return v
    return ""


def load_shapes(path):
    """与 hcpre_sim_app._parse_shapes_csv 同语义: b, bs|s, d|hidden (d 缺省 4096)"""
    shapes = []
    ndim = None
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            toks = [t.strip() for t in s.split(",") if t.strip() != ""]
            if not toks:
                continue
            try:
                vals = [int(t) for t in toks]
            except ValueError:
                continue  # header line
            if ndim is None:
                ndim = len(vals)
            if len(vals) >= 3:
                shapes.append((vals[0], vals[1], vals[2]))
            else:
                shapes.append((vals[0], vals[1], 4096))
    if not shapes:
        raise SystemExit(f"[perf-multi] no shapes parsed from {path}")
    return shapes


def load_op_rows(csv_path, name_filter):
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        name = _col(r, "Op Name", "Name", "Kernel Name") or "?"
        if name_filter and name_filter not in name:
            continue
        try:
            dur = float(_col(r, "Task Duration(us)", "Task Duration"))
        except (ValueError, TypeError):
            continue
        out.append((name, dur, _col(r, *SHAPE_COLS)))
    return out


def _shape_tokens(shape_str):
    """'1*512*4*4096;24*16384;...' -> first input's dims as ints"""
    first = shape_str.split(";")[0]
    return [int(t) for t in re.split(r"[*xX,\s]+", first) if t.strip().isdigit()]


def group_by_shape(op_rows, shapes):
    """-> per-shape sample lists (launch order) + strategy note"""
    n_shapes = len(shapes)
    # strategy 1: Input Shape column
    has_col = any(shp for _, _, shp in op_rows)
    if has_col:
        per = {i: [] for i in range(n_shapes)}
        unmatched = []
        for name, dur, shp in op_rows:
            dims = _shape_tokens(shp)
            hit = None
            for i, (b, bs, d) in enumerate(shapes):
                if dims == [b, bs, 4, d] or dims == [b * bs, 4, d]:
                    hit = i
                    break
            if hit is None:
                unmatched.append((name, shp))
            else:
                per[hit].append(dur)
        if all(len(v) > 0 for v in per.values()):
            return [per[i] for i in range(n_shapes)], "Input Shape column"
        if unmatched:
            print(f"[perf-multi] WARNING: {len(unmatched)} rows unmatched by Input Shape, "
                  f"falling back to positional; first: {unmatched[:2]}")
    # strategy 2: positional round-major (launch order: for round: for shape)
    durs = [d for _, d, _ in op_rows]
    if len(durs) % n_shapes != 0:
        print(f"[perf-multi] WARNING: {len(durs)} samples not divisible by {n_shapes} shapes; "
              f"trailing {len(durs) % n_shapes} rows dropped")
    per = [[] for _ in range(n_shapes)]
    n_rounds = len(durs) // n_shapes
    for r in range(n_rounds):
        for s in range(n_shapes):
            per[s].append(durs[r * n_shapes + s])
    return per, f"positional round-major ({n_rounds} rounds x {n_shapes} shapes)"


def print_stats(name, durs):
    n = len(durs)
    k = max(1, n // 5)
    head, tail = durs[:k], durs[-k:]
    mid = durs[k:n - k] if n - 2 * k > 0 else list(durs)
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


def main():
    if len(sys.argv) < 4:
        print("usage: perf_stats_multi.py <op_summary.csv> <shapes.csv> <outdir> [op_name_filter]")
        return 1
    op_csv, shapes_csv, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    name_filter = sys.argv[4] if len(sys.argv) > 4 else "HcPre"

    shapes = load_shapes(shapes_csv)
    op_rows = load_op_rows(op_csv, name_filter)
    if not op_rows:
        print(f"[perf-multi] no rows for filter '{name_filter}' in {op_csv}")
        return 1
    print(f"[perf-multi] {op_csv}: {len(op_rows)} samples, {len(shapes)} shapes")

    per, strategy = group_by_shape(op_rows, shapes)
    print(f"[perf-multi] grouping by: {strategy}")

    shape_dir = os.path.join(outdir, "perf_shape")
    os.makedirs(shape_dir, exist_ok=True)
    all_lines = []
    for (b, bs, d), durs in zip(shapes, per):
        tag = f"b={b} bs={bs} d={d}"
        print(f"--- shape {tag} ---")
        if not durs:
            print("  (no samples!)")
            continue
        print(f"  samples_1line   = " + ",".join(f"{v:.3f}" for v in durs))
        print_stats(tag, durs)
        vals_line = ",".join(f"{v:.4f}" for v in durs)
        all_lines.append((tag, vals_line))
        with open(os.path.join(shape_dir, f"b{b}_bs{bs}_d{d}.csv"), "w", encoding="utf-8") as f:
            f.write(f"# hc_pre perf shape b={b} bs={bs} d={d}: {len(durs)} samples (us), "
                    f"Task Duration, round-major launch order\n")
            f.write(vals_line + "\n")

    with open(os.path.join(outdir, "perf_multi_all.csv"), "w", encoding="utf-8") as f:
        f.write("# hc_pre multi-shape perf result: one line per shape, comma-separated\n")
        f.write("# Task Duration (us) samples in launch order (round-major: outer=iters, inner=shapes)\n")
        f.write("# NOTE: for perf_extreme_eval.py use the per-shape files under perf_shape/ "
                "(this file mixes shapes and is for record only)\n")
        for tag, line in all_lines:
            f.write(f"# shape {tag}\n")
            f.write(line + "\n")

    summary_path = os.path.join(outdir, "perf_multi_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"op_summary: {op_csv}\nshapes: {shapes_csv}\n"
                f"grouping: {strategy}\nsamples: {len(op_rows)}\n\n")
        for (b, bs, d), durs in zip(shapes, per):
            f.write(f"--- shape b={b} bs={bs} d={d} ---\n")
            if not durs:
                f.write("  (no samples!)\n")
                continue
            f.write("  samples_1line   = " + ",".join(f"{v:.3f}" for v in durs) + "\n")
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_stats(f"b={b} bs={bs} d={d}", durs)
            f.write(buf.getvalue())
    print(f"[perf-multi] results written: {summary_path}")
    print(f"[perf-multi] per-shape (perf_extreme_eval ready): {shape_dir}/b*_bs*_d*.csv")
    print(f"[perf-multi] combined record: {os.path.join(outdir, 'perf_multi_all.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
