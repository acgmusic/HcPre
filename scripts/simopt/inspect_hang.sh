#!/bin/bash
docker exec cann_container bash -c 'ls -la /root/HcPre/comb_dump.pt 2>/dev/null; echo ---; ls -t /root/HcPre/sim_dump_out/ 2>/dev/null | head -3'
