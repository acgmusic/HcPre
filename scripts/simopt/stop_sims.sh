#!/bin/bash
docker exec cann_container bash -c 'pkill -9 msprof; pkill -9 -f hcpre_sim_app; sleep 2; ps aux | grep -E "msprof|hcpre_sim" | grep -v grep | wc -l; rm -rf /root/HcPre/sim_out/* /root/HcPre/sim_dump_out; echo cleaned'
