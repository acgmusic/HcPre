#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""列出 veccore0 的 MTE2/VECTOR 队列事件序列(挂死现场), 找互等 flag 对"""
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
    evs = []
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
            if pv not in seg:
                continue
            ts = find_num(seg, b'"ts":')
            tid = find_str(seg, b'"tid":"')
            if ts is None or tid not in ('MTE2', 'VECTOR'):
                continue
            ph = find_str(seg, b'"ph":"')
            nm = find_str(seg, b'"name":"') or '?'
            eid = find_str(seg, b'"id":"')
            if ph == 'X' or (ph in ('B', 'E', 's', 't') and nm in ('SET_FLAG', 'WAIT_FLAG', 'flow')):
                evs.append((ts, tid, ph, nm, eid, find_str(seg, b'"detail":"')))
    evs.sort()
    print('total MTE2/VECTOR sync+copy events: %d' % len(evs))
    # 只打印 9.5us 之后(MTE2 最后活动 10.87 附近)与同步类事件
    print('== sync-relevant events after 9.0us ==')
    n = 0
    for ts, tid, ph, nm, eid, detail in evs:
        if ts < 9.0:
            continue
        if nm in ('SET_FLAG', 'WAIT_FLAG', 'flow') or 'MOV_SRC' in nm or 'VCONV' in nm:
            print('  %9.4f %-6s ph=%s %-20s id=%s' % (ts, tid, ph, nm, eid))
            n += 1
            if n > 90:
                print('  ... (truncated)')
                break


if __name__ == '__main__':
    main()
