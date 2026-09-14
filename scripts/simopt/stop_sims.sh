#!/bin/bash
docker exec cann_container bash -c 'rm -rf /root/HcPre/sim_out /root/HcPre/sim_dump_out; mkdir -p /root/HcPre/sim_out; echo 容器已清理'
