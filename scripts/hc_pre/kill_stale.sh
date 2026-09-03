#!/bin/bash
# Kill stale build processes (safe: excludes own processes via parent filtering)
set -u
C=${CANN_CONTAINER:-cann_container}
docker exec "$C" bash -c '
# only kill processes whose *executable* or *args* clearly belong to the old build,
# and never match our own command line: use precise patterns.
for pat in "^/usr/local/python3.11.15/bin/ninja" "cmake/data/bin/cmake --build" "ascend_protobuf_build" "^/usr/bin/gmake" "cc1plus" "/usr/bin/c\+\+ .*protobuf"; do
  pkill -9 -f "$pat" 2>/dev/null
done
sleep 1
echo "--- remaining build procs ---"
ps -eo pid,comm,args --no-headers | grep -E "ninja|gmake|cc1plus|protobuf_build" | grep -v grep | head -5 || echo "(clean)"
'
