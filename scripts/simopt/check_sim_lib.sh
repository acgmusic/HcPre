#!/bin/bash
docker exec cann_container bash -c '
LIB=/usr/local/Ascend/cann-9.1.0/x86_64-linux/simulator/Ascend910_9382/lib/libpem_davinci.so
echo "== nm 符号 =="; nm -D "$LIB" 2>/dev/null | grep -iE "vnchwconv|v4dtrans|ldva|va_reg" | head -20
echo "== strings =="; strings "$LIB" 2>/dev/null | grep -iE "vnchwconv|v4dtrans" | head -10
'
