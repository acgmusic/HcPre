#!/bin/bash
echo "=== WSL /root/HcPre link type ==="
ls -la /root/ 2>/dev/null | grep -i hcpre
echo "=== WSL /root/HcPre/sim_out NOW ==="
ls -la /root/HcPre/sim_out/ 2>&1 | head -6
echo "=== marker test: create in WSL ==="
touch /root/HcPre/XX_MARKER_FROM_WSL 2>/dev/null && echo "wsl touch ok"
echo "=== container sees marker? ==="
docker exec cann_container bash -c 'ls -la /root/HcPre/XX_MARKER_FROM_WSL 2>&1; ls -la /root/ | grep -i hcpre; stat -c "%F" /root/HcPre 2>/dev/null'
echo "=== container marker ==="
docker exec cann_container bash -c 'touch /root/HcPre/XX_MARKER_FROM_CONTAINER && echo "container touch ok"'
ls -la /root/HcPre/XX_MARKER_FROM_CONTAINER 2>&1
rm -f /root/HcPre/XX_MARKER_FROM_WSL /root/HcPre/XX_MARKER_FROM_CONTAINER 2>/dev/null
docker exec cann_container bash -c 'rm -f /root/HcPre/XX_MARKER_FROM_WSL /root/HcPre/XX_MARKER_FROM_CONTAINER' 2>/dev/null
echo done
