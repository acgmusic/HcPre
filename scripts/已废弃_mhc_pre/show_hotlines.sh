#!/bin/bash
# One-shot: locate busiest core code_exe.csv in sim_out_mhc and print hot source lines
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}
docker exec "$C" bash -c '
D=/root/HcPre/sim_out_mhc
CSV=$(find "$D" -path "*core0.veccore0*code_exe.csv" | head -1)
echo "csv: $CSV"
python3 - "$CSV" <<"PYEOF"
import csv, os, re, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    r["cycles"] = int(r["cycles"])
rows.sort(key=lambda r: r["cycles"], reverse=True)
total = sum(r["cycles"] for r in rows) or 1
print("=== hot source lines on core0.veccore0 ===")
for r in rows[:12]:
    m = re.match(r"^(.*):(\d+)$", r["code"])
    loc = (os.path.basename(m.group(1)) + ":" + m.group(2)) if m else r["code"]
    pct = r["cycles"] / total * 100
    print("  %-46s %9d cyc (%5.1f%%) calls=%s" % (loc, r["cycles"], pct, r["call_count"]))
PYEOF
'
