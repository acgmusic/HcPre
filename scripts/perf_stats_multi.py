#!/usr/bin/env python3
"""Multi-round, multi-iter, multi-shape msprof op_summary aggregation for hc_pre.

Usage:
    python3 perf_stats_multi.py <shapes.csv> <outdir> <op_summary_r1.csv> \
        [<op_summary_r2.csv> ...] [--filter HcPre]

Input model (matches run.sh `perf <shapes.csv> [rounds]`):
  * each ROUND is one independent msprof process invocation (fresh state)
  * inside a round the app makes HC_PRE_ITERS passes over all shapes
    (iter-major: outer=iters, inner=shapes)
  * argument order of the op_summary csvs = round order (run.sh passes them
    as perf_out_multi/round_001..N/op_summary_*.csv)

For every shape this builds a rounds x iters sample matrix (row r, col i =
round r's i-th in-process launch of that shape) and:
  * prints, per shape: the full time-series (one line per round, iters
    comma-separated values) + statistics over all rounds*iters samples
  * writes <outdir>/perf_multi_summary.txt   (full human-readable summary)
  * writes <outdir>/perf_multi_all.csv       (all shapes' matrices, '#' comments)
  * writes <outdir>/perf_shape/b{b}_bs{bs}_d{d}.csv  (THE per-shape input for
    perf_extreme_eval.py: rounds lines x iters values — 行=独立进程块,
    列=进程内采样, exactly the t[t,n] block format the project expects)

Within a round, rows are matched to shapes primarily via the op_summary
"Input Shape" column (both 4D "b*bs*4*d" and folded "bs*4*d" forms);
fallback = positional (launch order inside the round: iter-major stride).
Rounds whose per-shape sample counts differ from the majority are skipped
with a warning, keeping the matrix rectangular.
"""

import argparse
import csv
import os
import re
import statistics
from collections import Counter

SHAPE_COLS = ("Input Shape", "Input Shapes", "Shapes", "InputShapes")


def _col(row, *names):
    for n in names:
        v = row.get(n)
        if v:
            return v
    return ""


def load_shapes(path):
    """与 hcpre_sim_app._parse_shapes_csv 同语义: b, bs|s, d|hidden (d 缺省 4096);
    拒绝重复 shape(否则 per-shape 输出文件相互覆盖)"""
    shapes = []
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
            if len(vals) >= 3:
                shape = (vals[0], vals[1], vals[2])
            elif len(vals) == 2:
                shape = (vals[0], vals[1], 4096)
            else:
                raise SystemExit(f"[perf-multi] bad shape row (need b,bs[,d]): {s!r}")
            if shape in shapes:
                raise SystemExit(f"[perf-multi] duplicate shape b={shape[0]} bs={shape[1]} "
                                 f"d={shape[2]} in {path} (per-shape files would collide)")
            shapes.append(shape)
    if not shapes:
        raise SystemExit(f"[perf-multi] no shapes parsed from {path}")
    return shapes


def load_op_rows(csv_path, name_filter):
    """-> [(dur, input_shape_str)] 只保留名字匹配且有 Task Duration 的行"""
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
        out.append((dur, _col(r, *SHAPE_COLS)))
    return out


def _shape_tokens(shape_str):
    """'1*512*4*4096;24*16384;...' -> first input's dims as ints"""
    first = shape_str.split(";")[0]
    return [int(t) for t in re.split(r"[*xX,\s]+", first) if t.strip().isdigit()]


def round_to_shapes(op_rows, shapes, round_tag, allow_shape_col=True):
    """一轮的行 -> {shape_idx: [dur 按进程内 iter 序]} 或 None(该轮不可信, 跳过)

    一轮内行序 = 发射序 = iter-major(外层 iter, 内层 shape):
    shape s 的第 i 个采样在第 (i-1)*nShapes + s 行。
    allow_shape_col=False 时直接走位置映射(折叠形歧义等场景)。
    """
    n_shapes = len(shapes)
    has_col = any(shp for _, shp in op_rows)
    if has_col and allow_shape_col:
        per = {i: [] for i in range(n_shapes)}
        for dur, shp in op_rows:
            dims = _shape_tokens(shp)
            for i, (b, bs, d) in enumerate(shapes):
                if dims == [b, bs, 4, d] or dims == [b * bs, 4, d]:
                    per[i].append(dur)
                    break
        counts = {len(v) for v in per.values()}
        if len(counts) == 1 and next(iter(counts)) > 0:
            return per
        print(f"[perf-multi] WARNING round {round_tag}: Input Shape matched "
              f"{counts or 'nothing'} (shapes={n_shapes}), trying positional")
    # positional: iter-major stride 映射
    durs = [d for d, _ in op_rows]
    if len(durs) % n_shapes != 0 or len(durs) == 0:
        print(f"[perf-multi] WARNING round {round_tag}: {len(durs)} HcPre rows not "
              f"divisible by {n_shapes} shapes, round SKIPPED")
        return None
    per = {i: durs[i::n_shapes] for i in range(n_shapes)}
    return per


def print_stats(flat, n_rounds, n_iters):
    n = len(flat)
    k = max(1, n // 5)
    head, tail = flat[:k], flat[-k:]
    mid = flat[k:n - k] if n - 2 * k > 0 else list(flat)
    print(f"  rounds x iters  = {n_rounds} x {n_iters} (total {n})")
    print(f"  min             = {min(flat):.3f} us")
    print(f"  max             = {max(flat):.3f} us")
    print(f"  mean            = {statistics.mean(flat):.3f} us")
    print(f"  median (P50)    = {statistics.median(flat):.3f} us")
    if n > 1:
        print(f"  stddev          = {statistics.stdev(flat):.3f} us")
    print(f"  head avg        = {statistics.mean(head):.3f} us  (first {k})")
    print(f"  tail avg        = {statistics.mean(tail):.3f} us  (last {k})")
    print(f"  mid avg         = {statistics.mean(mid):.3f} us  (middle {len(mid)})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("shapes", help="shapes csv (b,bs|s[,d|hidden])")
    p.add_argument("outdir", help="output directory")
    p.add_argument("op_csvs", nargs="+", help="op_summary csvs, one per round, in round order")
    p.add_argument("--filter", default="HcPre", help="op name filter (default HcPre)")
    args = p.parse_args()

    shapes = load_shapes(args.shapes)
    n_shapes = len(shapes)
    # 折叠形歧义检测: (b*bs, d) 相同的不同 shape 在 op_summary 折叠形中不可区分
    # (如 (1,512) 与 (2,256) 均折叠为 512*4*4096) -> 禁用 Input Shape 匹配, 全程位置映射
    fold_keys = [(b * bs, d) for (b, bs, d) in shapes]
    allow_shape_col = len(set(fold_keys)) == len(fold_keys)
    if not allow_shape_col:
        print("[perf-multi] NOTE: shapes csv contains distinct (b,bs) with same b*bs "
              "product; Input Shape column cannot disambiguate -> positional mapping only")
    # 每轮每 shape 的采样序列(带原始轮号)
    round_series = []  # [(orig_round_no, {shape_idx: [iters durs]}), ...]
    n_ok = 0
    for r, op_csv in enumerate(args.op_csvs, 1):
        op_rows = load_op_rows(op_csv, args.filter)
        if not op_rows:
            print(f"[perf-multi] WARNING round {r}: no '{args.filter}' rows in {op_csv}, SKIPPED")
            continue
        per = round_to_shapes(op_rows, shapes, r, allow_shape_col)
        if per is None:
            continue
        round_series.append((r, per))
        n_ok += 1
    if not round_series:
        raise SystemExit("[perf-multi] no usable rounds collected")

    # 迭代次数一致性: 取多数轮的每 shape 采样数为 iters, 不一致的轮剔除(保持矩阵矩形)
    iters_counts = Counter(len(per[0]) for _, per in round_series)
    n_iters = iters_counts.most_common(1)[0][0]
    dropped = {rn for rn, per in round_series if len(per[0]) != n_iters}
    for rn in sorted(dropped):
        print(f"[perf-multi] WARNING round {rn}: iters mismatch != majority {n_iters}, round DROPPED")
    round_series = [(rn, per) for rn, per in round_series if rn not in dropped]
    n_rounds = len(round_series)
    print(f"[perf-multi] {n_shapes} shapes, {n_ok}/{len(args.op_csvs)} rounds collected, "
          f"{n_rounds} kept x {n_iters} iters")

    # 矩阵: matrix[i] = [round_1_series, round_2_series, ...] (每元素为该轮 iters 个采样)
    matrix = {i: [] for i in range(n_shapes)}
    for _, per in round_series:
        for i in range(n_shapes):
            matrix[i].append(per[i])

    shape_dir = os.path.join(args.outdir, "perf_shape")
    os.makedirs(shape_dir, exist_ok=True)
    for i, (b, bs, d) in enumerate(shapes):
        rows = matrix[i]
        tag = f"b={b} bs={bs} d={d}"
        print(f"--- shape {tag} ---")
        for r_i, row in enumerate(rows, 1):
            print(f"  round{r_i:03d}        = " + ",".join(f"{v:.3f}" for v in row))
        flat = [v for row in rows for v in row]
        print_stats(flat, n_rounds, n_iters)
        with open(os.path.join(shape_dir, f"b{b}_bs{bs}_d{d}.csv"), "w", encoding="utf-8") as f:
            f.write(f"# hc_pre perf shape b={b} bs={bs} d={d}: {n_rounds} rounds (independent"
                    f" msprof processes) x {n_iters} in-process samples (us), Task Duration\n")
            f.write("# format: one line per round, comma-separated in-iter samples "
                    "(perf_extreme_eval block format: row=process block, col=sample)\n")
            for row in rows:
                f.write(",".join(f"{v:.4f}" for v in row) + "\n")

    with open(os.path.join(args.outdir, "perf_multi_all.csv"), "w", encoding="utf-8") as f:
        f.write("# hc_pre multi-shape perf result: per shape, rounds lines x iters values\n")
        f.write("# Task Duration (us); round = independent msprof process, iter = in-process pass\n")
        f.write("# NOTE: for perf_extreme_eval.py use the per-shape files under perf_shape/\n")
        for i, (b, bs, d) in enumerate(shapes):
            f.write(f"# shape b={b} bs={bs} d={d}\n")
            for row in matrix[i]:
                f.write(",".join(f"{v:.4f}" for v in row) + "\n")

    summary_path = os.path.join(args.outdir, "perf_multi_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        import contextlib
        import io
        f.write(f"shapes: {args.shapes}\nrounds kept: {n_rounds} (collected {n_ok}/"
                f"{len(args.op_csvs)}), iters/round: {n_iters}\n"
                f"sampling: per round = independent msprof process, in-process iters passes "
                f"over shapes\n\n")
        for i, (b, bs, d) in enumerate(shapes):
            f.write(f"--- shape b={b} bs={bs} d={d} ---\n")
            for r_i, row in enumerate(matrix[i], 1):
                f.write(f"  round{r_i:03d}        = " + ",".join(f"{v:.3f}" for v in row) + "\n")
            flat = [v for row in matrix[i] for v in row]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_stats(flat, n_rounds, n_iters)
            f.write(buf.getvalue())
    print(f"[perf-multi] results written: {summary_path}")
    print(f"[perf-multi] per-shape (perf_extreme_eval ready): {shape_dir}/b*_bs*_d*.csv")
    print(f"[perf-multi] combined record: {os.path.join(args.outdir, 'perf_multi_all.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
