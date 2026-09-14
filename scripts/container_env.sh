# Sourced inside the container before any build/sim command.
# Pins CANN 9.1.0 and cleans the env pollution baked into the container image
# (the image ships CANN 9.0.0-beta.1 env vars pointing at stale paths).
unset PYTHONPATH CMAKE_PREFIX_PATH TOOLCHAIN_HOME ASCEND_AICPU_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
source $ASCEND_HOME_PATH/set_env.sh
export PATH=/usr/local/python3.11.15/bin:$PATH
# x86_64-linux/lib64(+:device/lib64) hosts libascendcl/libascend_hal needed by torch_npu
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64/device/lib64:$LD_LIBRARY_PATH
