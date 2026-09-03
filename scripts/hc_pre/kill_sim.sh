#!/bin/bash
# Kill the running msprof simulation process group in the container
set -u
C=${CANN_CONTAINER:-cann_container}
docker exec "$C" bash -c '
echo "--- sim processes before kill ---"
ps -eo pid,comm,args --no-headers | grep -iE "msprof|msopprof|hcpre_sim|TmSim|pem_davinci|camodel|simulator" | grep -v grep | head -10
# kill msopprof + python sim app + simulator model libs
pkill -9 -f msopprof 2>/dev/null
pkill -9 -f hcpre_sim_app 2>/dev/null
pkill -9 -f "msprof" 2>/dev/null
pkill -9 -f "libpem_davinci" 2>/dev/null
pkill -9 -f "libmodel_top" 2>/dev/null
pkill -9 -f "libstars" 2>/dev/null
sleep 1
echo "--- remaining ---"
ps -eo pid,comm,args --no-headers | grep -iE "msprof|msopprof|hcpre_sim|simulator" | grep -v grep | head -5 || echo "(clean)"
'
