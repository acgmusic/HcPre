#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""挂死定点: veccore0 最后 20 条 SCALAR/FLOWCTRL 事件的 pc + 记录4 的源码映射"""
import sys, os, re, json

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

    pv = b'"pid":"core0.veccore0"'
    tail_ev = []   # (ts, tid, name, pc) — keep last N of SCALAR/FLOWCTRL
    carry = b''
    N = 4000
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
            if pv not in seg:
                continue
            ts = find_num(seg, b'"ts":')
            tid = find_str(seg, b'"tid":"')
            if ts is None or tid not in ('SCALAR', 'FLOWCTRL'):
                continue
            if find_str(seg, b'"ph":"') != 'X':
                continue
            tail_ev.append((ts, tid, find_str(seg, b'"name":"'), find_str(seg, b'"pc_addr":"')))
    tail_ev.sort()
    print('total SCALAR/FLOWCTRL events: %d' % len(tail_ev))
    print('== last 20 ==')
    for ts, tid, nm, pc in tail_ev[-20:]:
        print('  %9.3f %-8s %-16s pc=%s' % (ts, tid, nm, pc))

    # 记录4: pc -> (Source, AscendC Inner Code)
    f2 = open(path, 'rb')
    size = os.path.getsize(path)
    # reuse layout: t1 known; find rec4 start: scan from t1 for first '{' after binary run
    f2.seek(t1)
    # records 2..4 follow; simplest: search from end for record-4 by scanning back for '{"Cores"' pattern
    chunk = 64 * 1024 * 1024
    pos4 = None
    end = size
    while end > t1:
        start = max(t1, end - chunk)
        f2.seek(start)
        buf = f2.read(end - start)
        i = buf.rfind(b'{"Cores"')
        if i != -1:
            pos4 = start + i
            break
        end = start + 10
    if pos4 is None:
        print('rec4 not found')
        return
    f2.seek(pos4)
    dec = json.JSONDecoder()
    r4 = dec.raw_decode(f2.read(size - pos4).decode('utf-8', 'replace'))[0]
    pc_map = {}
    for e in r4['Instructions']:
        pc_map[e['Address']] = (e.get('Source', ''), e.get('AscendC Inner Code', ''))
    print('\n== pc -> source ==')
    seen = []
    for ts, tid, nm, pc in tail_ev[-20:]:
        if not pc or pc in seen:
            continue
        seen.append(pc)
        src, inner = pc_map.get(pc, ('?', ''))
        print('  %-10s %-14s | %-44s | %s' % (pc, nm, src[:44], inner.replace('\n', ' | ')[:120]))


if __name__ == '__main__':
    main()
