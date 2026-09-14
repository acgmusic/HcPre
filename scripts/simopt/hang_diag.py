#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""挂死诊断: 每条流水的最后事件 + 未完成的长 WAIT"""
import sys, os, re

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


def main():
    path = sys.argv[1]
    f = open(path, 'rb')
    f.seek(12)
    assert f.read(18) == b'{"displayTimeUnit"'
    pos, carry = 12, b''
    size = os.path.getsize(path)
    while True:
        f.seek(pos)
        buf = f.read(16 * 1024 * 1024)
        if not buf:
            raise SystemExit('no rec1 end')
        data = carry + buf
        m = BINRUN.search(data)
        if m:
            t1 = pos - len(carry) + m.start()
            break
        carry = data[-2:]
        pos += len(buf)
    f.seek(12)
    head = f.read(4096)
    pos = 12 + head.find(b'"traceEvents":[') + len(b'"traceEvents":[')

    targets = [b'"pid":"core0.veccore0"', b'"pid":"core0.veccore1"', b'"pid":"core0.cubecore0"']
    last = {t: {} for t in targets}   # tid -> (ts, dur, name, ph, id)
    waits = {t: [] for t in targets}
    gmax = -1e18
    carry = b''
    while pos < t1:
        buf = f.read(min(16 * 1024 * 1024, t1 - pos))
        if not buf:
            break
        pos += len(buf)
        data = carry + buf
        lastsep = data.rfind(SEP)
        if lastsep == -1:
            carry = data
            continue
        carry = data[lastsep + len(SEP):]
        for seg in data[:lastsep].split(SEP):
            ts = find_num(seg, b'"ts":')
            if ts is None:
                continue
            dur = find_num(seg, b'"dur":') or 0.0
            if ts + dur > gmax:
                gmax = ts + dur
            for t in targets:
                if t in seg:
                    tid = find_str(seg, b'"tid":"')
                    nm = find_str(seg, b'"name":"') or '?'
                    ph = find_str(seg, b'"ph":"')
                    eid = find_str(seg, b'"id":"')
                    if tid is None:
                        continue
                    prev = last[t].get(tid)
                    if prev is None or ts >= prev[0]:
                        last[t][tid] = (ts, dur, nm, ph, eid)
                    if 'WAIT' in nm and dur > 1.0:
                        waits[t].append((ts, dur, nm, tid))
    print('global last event end: %.3f us' % gmax)
    for t in targets:
        print('\n=== %s ===' % t.decode())
        for tid, (ts, dur, nm, ph, eid) in sorted(last[t].items(), key=lambda x: -x[1][0]):
            print('  last %-8s %9.3f dur=%8.3f %-18s ph=%s id=%s' % (tid, ts, dur, nm, ph, eid))
        ws = sorted(waits[t], key=lambda x: -x[1])[:4]
        if ws:
            print('  -- longest WAITs --')
            for ts, dur, nm, tid in ws:
                print('     %9.3f dur=%8.3f %-16s on %s' % (ts, dur, nm, tid))


if __name__ == '__main__':
    main()
