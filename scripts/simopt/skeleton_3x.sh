#!/bin/bash
# 骨架(双关) 3 连跑统计崩溃率; 每次给足超时让 dump 落盘
for i in 1 2 3; do
  docker exec cann_container bash -c "rm -rf /root/HcPre/sim_out/* /root/HcPre/sim_dump_out /root/HcPre/comb_dump.pt /tmp/dbi_run.log; timeout 1200 bash /root/HcPre/scripts/simopt/check_state.sh > /tmp/dbi_run.log 2>&1; true"
  docker exec cann_container bash -c "
    echo \"run$i: DBI=\$(grep -c 'DBI' /tmp/dbi_run.log) Model=\$(grep -oE 'Model RUN TIME: [0-9.e+]+' /tmp/dbi_run.log | head -1) dump=\$(ls /root/HcPre/comb_dump.pt 2>/dev/null && echo YES || echo NO)\""
done
