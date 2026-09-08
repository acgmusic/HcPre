# Sourced before any build/sim command (works in any environment).
#
# Goals:
#   1. If the environment already has a working CANN (ASCEND_HOME_PATH set and
#      valid, e.g. an official Ascend container image), keep it as-is.
#   2. Otherwise, auto-detect the newest installed CANN under /usr/local/Ascend
#      and source its set_env.sh.
#   3. Clean common env pollution that breaks builds (stale PYTHONPATH etc.)
#      only when we are switching CANN versions ourselves.
#
# Environment overrides:
#   ASCEND_HOME_PATH  if already set and valid, it wins (no change)
#   CANN_VERSION      force a specific version dir, e.g. CANN_VERSION=9.1.0

# --- 1. already-valid environment? ---
_env_ok() {
    [ -n "${ASCEND_HOME_PATH:-}" ] || return 1
    [ -e "${ASCEND_HOME_PATH}/set_env.sh" ] || return 1
    return 0
}

if ! _env_ok; then
    # --- 2. auto-detect ---
    ASCEND_ROOT=${ASCEND_ROOT:-/usr/local/Ascend}

    if [ -n "${CANN_VERSION:-}" ]; then
        # forced version
        CAND="${ASCEND_ROOT}/cann-${CANN_VERSION}"
        if [ -e "${CAND}/set_env.sh" ]; then
            export ASCEND_HOME_PATH="$CAND"
        else
            echo "[env] CANN_VERSION=${CANN_VERSION} not found at ${CAND}" >&2
            return 1 2>/dev/null || exit 1
        fi
    elif [ -L "${ASCEND_ROOT}/cann" ]; then
        # 'cann' symlink exists (standard multi-version layout) -> follow it
        export ASCEND_HOME_PATH="$(readlink -f "${ASCEND_ROOT}/cann")"
    else
        # pick the newest cann-* dir
        CAND=$(ls -d "${ASCEND_ROOT}"/cann-* 2>/dev/null | sort -V | tail -1)
        if [ -n "$CAND" ] && [ -e "$CAND/set_env.sh" ]; then
            export ASCEND_HOME_PATH="$CAND"
        elif [ -e "${ASCEND_ROOT}/ascend-toolkit/latest/set_env.sh" ]; then
            # toolkit layout
            export ASCEND_HOME_PATH="${ASCEND_ROOT}/ascend-toolkit/latest"
        else
            echo "[env] no CANN installation found under ${ASCEND_ROOT}" >&2
            return 1 2>/dev/null || exit 1
        fi
    fi

    # clean pollution only when we manage the switch
    unset PYTHONPATH CMAKE_PREFIX_PATH TOOLCHAIN_HOME ASCEND_AICPU_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME
    source "${ASCEND_HOME_PATH}/set_env.sh"
fi

# --- 3. common additions (idempotent, safe in both cases) ---
# libascendcl/libascend_hal for x86 hosts live under x86_64-linux/lib64
if [ -d "${ASCEND_HOME_PATH}/x86_64-linux/lib64" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${ASCEND_HOME_PATH}/x86_64-linux/lib64:"*) ;;
        *) export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/x86_64-linux/lib64:${ASCEND_HOME_PATH}/x86_64-linux/lib64/device/lib64:${LD_LIBRARY_PATH:-}" ;;
    esac
fi

echo "[env] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

# --- 4. merge_obj_text.sh idempotency guard (self-healing, any arch) ---
# The in-place ld.lld text merge in CANN's legacy_modules is not idempotent and
# breaks no-change incremental pybind rebuilds ("unknown file type"). The
# patched copy lives in the workspace (cann-9.1.0/x86_64-linux/.../merge_obj_text.sh,
# marked HC_MERGE_GUARD; the script itself is arch-independent); apply it to the
# active CANN install (x86_64-linux docker or aarch64-linux real machine) when missing.
_merge_src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/cann-9.1.0/x86_64-linux/tikcpp/ascendc_kernel_cmake/legacy_modules/util/merge_obj_text.sh"
if [ -f "${_merge_src}" ]; then
    for _arch in x86_64-linux aarch64-linux; do
        _merge_dst="${ASCEND_HOME_PATH}/${_arch}/tikcpp/ascendc_kernel_cmake/legacy_modules/util/merge_obj_text.sh"
        if [ -f "${_merge_dst}" ] && ! grep -q "HC_MERGE_GUARD" "${_merge_dst}" 2>/dev/null; then
            if cp "${_merge_src}" "${_merge_dst}" 2>/dev/null; then
                echo "[env] applied HC_MERGE_GUARD patch to ${_merge_dst}"
            else
                echo "[env] WARN: ${_merge_dst} lacks the HC_MERGE_GUARD patch and is not writable" >&2
                echo "[env] WARN: incremental pybind rebuild will fail until you run:" >&2
                echo "[env] WARN:   sudo cp ${_merge_src} ${_merge_dst}" >&2
            fi
        fi
    done
fi
