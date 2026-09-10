#!/bin/bash
# 串行模式 check_state: 预置 parsim=0 的 config_stars.json 绕过 24 线程竞态 DBI 风暴
set -e
docker start cann_container >/dev/null
docker exec cann_container bash -c '
CFG=/root/HcPre/simopt_serial_cfg
mkdir -p $CFG
cat > $CFG/config_stars.json <<EOF
{
    "stars": {
        "ffts_mode": 1
    },
    "model_top": {
        "sim_type": 0,
        "num_aic": 24,
        "num_aiv": 48
    },
    "pem": {
        "parsim": 0,
        "parsim_thd_limit": 1
    }
}
EOF
echo "serial config ready:"
cat $CFG/config_stars.json
'
