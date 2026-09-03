"""Summarize msprof op simulator output: per-core pipe utilization (instr_exe.csv)
and hot source lines (code_exe.csv). Usage: python3 parse_sim_csv.py <prof_out_dir>"""

import csv
import glob
import os
import re
import sys


def load_rows(path):
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def main(root):
    instr_files = sorted(glob.glob(os.path.join(root, "**", "*_instr_exe.csv"), recursive=True))
    code_files = sorted(glob.glob(os.path.join(root, "**", "*_code_exe.csv"), recursive=True))
    if not instr_files:
        print(f"no *_instr_exe.csv under {root}")
        return 1

    # rank cores by total cycles so the busiest cores get printed
    stats = []
    for f in instr_files:
        rows = load_rows(f)
        total = sum(int(r["cycles"]) for r in rows)
        stats.append((total, f, rows))
    stats.sort(reverse=True, key=lambda s: s[0])

    print(f"found {len(instr_files)} cores with instr_exe.csv; showing top "
          f"{min(8, len(stats))} by total cycles\n")
    for total, f, rows in stats[:8]:
        core = os.path.basename(os.path.dirname(f))
        pipe = {}
        for r in rows:
            pipe[r["pipe"]] = pipe.get(r["pipe"], 0) + int(r["cycles"])
        print(f"=== {core}  total={total} cycles ===")
        for p, c in sorted(pipe.items(), key=lambda x: -x[1])[:6]:
            print(f"    {p:<14} {c / total * 100:6.1f}%")
        rows.sort(key=lambda r: int(r["cycles"]), reverse=True)
        for r in rows[:3]:
            print(f"    top: {r['instr']} pipe={r['pipe']} calls={r['call_count']} cycles={r['cycles']}")
        print()

    # hot source lines for the busiest core only
    busiest = os.path.basename(os.path.dirname(stats[0][1]))
    for f in code_files:
        if os.path.basename(os.path.dirname(f)) == busiest:
            rows = load_rows(f)
            for r in rows:
                r["cycles"] = int(r["cycles"])
                r["call_count"] = int(r["call_count"])
            rows.sort(key=lambda r: r["cycles"], reverse=True)
            total = sum(r["cycles"] for r in rows) or 1
            print(f"=== hot source lines on {busiest} (code_exe) ===")
            for r in rows[:10]:
                m = re.match(r"^(.*):(\d+)$", r["code"])
                loc = f"{os.path.basename(m.group(1))}:{m.group(2)}" if m else r["code"]
                print(f"    L{loc:<40} {r['cycles']:>10} cyc ({r['cycles'] / total * 100:5.1f}%) "
                      f"calls={r['call_count']}")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
