#!/bin/bash
# Foreground bs=48 debug sim (most reliable path). Also regenerates the runner
# via docker exec -i (heredoc over stdin, which is the proven-working pattern).
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}
BS=${1:-48}

# write runner via -i stdin
docker exec -i "$C" bash > /dev/null <<EOS
cat > /root/run_mhcs_sim_fg.sh <<'RUN'
#!/bin/bash
source /root/HcPre/scripts/container_env.sh
V=/root/HcPre/mhc_pre_install/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH=\$V
export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend910_9382/lib:\$V/op_api/lib:\$LD_LIBRARY_PATH
cd /root/HcPre/ops-transformer/build
rm -rf /root/HcPre/sim_out_mhcs
msprof op simulator \\
  --application="\$PWD/test_aclnn_mhc_pre_sinkhorn_hcshape mhc_input_bs${BS}.bin /root/HcPre/mhcs_output_bs${BS}.bin" \\
  --output=/root/HcPre/sim_out_mhcs \\
  --kernel-name=MhcPreSinkhorn \\
  --launch-count=1 \\
  --soc-version=Ascend910_9382 \\
  --timeout=120
echo "MSPROF_RC=\$?"
RUN
chmod +x /root/run_mhcs_sim_fg.sh
EOS

# sanity: runner file non-empty
sz=$(docker exec "$C" stat -c%s /root/run_mhcs_sim_fg.sh)
echo "[fg-sim] runner size: $sz"
[ "$sz" -gt 100 ] || { echo "runner write failed"; exit 1; }

# ensure input bin exists in build dir
docker exec "$C" bash -c "cp -f /root/HcPre/mhc_input_bs${BS}.bin /root/HcPre/ops-transformer/build/ && ls -la /root/HcPre/ops-transformer/build/mhc_input_bs${BS}.bin"

# run FOREGROUND
docker exec -i "$C" /root/run_mhcs_sim_fg.sh 2>&1 | grep -aE "mhcs-ex|Model Stop|Start parse|Profiling results|Core operator|signal|MSPROF_RC|visualize" | head -15

echo "--- products ---"
docker exec -i "$C" bash <<'INNER'
chmod -R a+rX /root/HcPre/sim_out_mhcs 2>/dev/null
echo "instr_csv: $(find /root/HcPre/sim_out_mhcs -name '*_instr_exe.csv' 2>/dev/null | wc -l)"
echo "trace_json: $(find /root/HcPre/sim_out_mhcs -name 'trace.json' 2>/dev/null | wc -l)"
echo "visualize: $(find /root/HcPre/sim_out_mhcs -name 'visualize_data.bin' -exec ls -la {} \; 2>/dev/null)"
ls -la /root/HcPre/mhcs_output_bs${BS}.bin 2>/dev/null || echo "no output bin"
INNER
