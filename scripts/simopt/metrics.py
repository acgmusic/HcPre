#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从最新 OPPROF*/simulator/visualize_data.bin 提取关键流水指标（纯 stdlib, 容器内运行）"""
import sys, os, re, glob

MAGIC = bytes.fromhex('a8044f14')
SEP = b'},{'
BINRUN = re.compile(rb'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]{3,}')


def find_num(seg, key):
    i = seg.find(key)
    if i == -1:
        return None
    j = seg.find(b',', i)
    if j == -1:
        j = len(seg)
    try:
        return float(seg[i + len(key):j])
    except ValueError:
        return None


def find_str(seg, key):
    i = seg.find(key)
    if i == -1:
        return None
    j = seg.find(b'"', i + len(key))
    return None if j == -1 else seg[i + len(key):j].decode('utf-8', 'replace')


def scan_trace_range(f):
    f.seek(12)
    if f.read(18) != b'{"displayTimeUnit"':
        f.seek(0)
        head = f.read(64)
        raise SystemExit('record 1 (traceEvents JSON) not found at offset 12: ' + head[:32].hex())
    pos, carry = 12, b''
    while True:
        f.seek(pos)
        buf = f.read(16 * 1024 * 1024)
        if not buf:
            raise SystemExit('rec1 end not found')
        data = carry + buf
        m = BINRUN.search(data)
        if m:
            return 12, pos - len(carry) + m.start()
        carry = data[-2:]
        pos += len(buf)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '/root/HcPre/sim_out'
    cands = sorted(glob.glob(os.path.join(root, 'OPPROF*', 'simulator', 'visualize_data.bin')),
                   key=os.path.getmtime)
    if not cands:
        raise SystemExit('no visualize_data.bin under ' + root)
    path = cands[-1]
    print('BIN %s (%.2f GB)' % (path, os.path.getsize(path) / 1024 ** 3))
    f = open(path, 'rb')
    t0, t1 = scan_trace_range(f)
    f.seek(t0)
    head = f.read(4096)
    pos = t0 + head.find(b'"traceEvents":[') + len(b'"traceEvents":[')

    gmin, gmax = 1e18, -1e18
    pv = b'"pid":"core0.veccore0"'
    pc = b'"pid":"core0.cubecore0"'
    v_mte2, v_mte3, v_wait, v_tids = [], [], [], {}
    v_scalar_first = None
    c_l1_first, c_l1_n, c_mmad = None, 0, 0.0
    carry = b''
    while pos < t1:
        buf = f.read(min(16 * 1024 * 1024, t1 - pos))
        if not buf:
            break
        pos += len(buf)
        data = carry + buf
        last = data.rfind(SEP)
        if last == -1:
            carry = data
            continue
        carry = data[last + len(SEP):]
        for seg in data[:last].split(SEP):
            if b'"ph":"X"' not in seg:
                continue
            ts = find_num(seg, b'"ts":')
            if ts is None:
                continue
            dur = find_num(seg, b'"dur":') or 0.0
            if ts < gmin:
                gmin = ts
            if ts + dur > gmax:
                gmax = ts + dur
            if pv in seg:
                tid = find_str(seg, b'"tid":"')
                name = find_str(seg, b'"name":"') or ''
                v_tids[tid] = v_tids.get(tid, 0.0) + dur
                if tid == 'MTE2' and 'MOV_SRC_TO_DST' in name:
                    v_mte2.append((ts, dur))
                elif tid == 'MTE3' and 'MOV_SRC_TO_DST' in name:
                    v_mte3.append((ts, dur))
                elif tid == 'FLOWCTRL' and 'WAIT' in name:
                    v_wait.append((ts, dur))
                elif tid == 'SCALAR' and v_scalar_first is None:
                    v_scalar_first = ts
            elif pc in seg:
                tid = find_str(seg, b'"tid":"')
                name = find_str(seg, b'"name":"') or ''
                if tid == 'MTE2' and 'MOV_OUT_TO_L1' in name:
                    c_l1_n += 1
                    if c_l1_first is None:
                        c_l1_first = ts
                elif tid == 'CUBE' and name == 'MMAD':
                    c_mmad += dur

    span = gmax - gmin
    print('WALL_SPAN_US %.3f  (ev span %.3f..%.3f)' % (span, gmin, gmax))
    busy2 = sum(d for _, d in v_mte2)
    busy3 = sum(d for _, d in v_mte3)
    print('V0 scalar_first %.3f | MTE2 copy: n=%d first=%.3f busy=%.2f | MTE3 copy: n=%d first=%.3f busy=%.2f max=%.3f'
          % (v_scalar_first if v_scalar_first is not None else -1,
             len(v_mte2), v_mte2[0][0] if v_mte2 else -1, busy2,
             len(v_mte3), v_mte3[0][0] if v_mte3 else -1, busy3,
             max((d for _, d in v_mte3), default=0)))
    gaps = [(b[0] - a[0] - a[1], a[0]) for a, b in zip(v_mte2, v_mte2[1:])]
    idle2 = sum(g for g, _ in gaps if g > 0)
    print('V0 MTE2 idle_between %.3f us | top gaps: %s'
          % (idle2, sorted(['%.2f@%.1f' % (g, t) for g, t in gaps if g > 0.05], reverse=True)[:6]))
    print('V0 crosscore_wait n=%d total=%.3f | early(<6): %s'
          % (len(v_wait), sum(d for _, d in v_wait),
             ['%.2f+%.2f' % (t, d) for t, d in v_wait if t < 6][:5]))
    print('V0 tid busy: %s' % {k: round(v, 1) for k, v in sorted(v_tids.items(), key=lambda x: -x[1])[:5]})
    print('C0 first_L1_read %.3f n=%d | MMAD busy %.2f' % (c_l1_first if c_l1_first is not None else -1, c_l1_n, c_mmad))


if __name__ == '__main__':
    main()
