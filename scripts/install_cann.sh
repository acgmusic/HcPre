#!/bin/bash
# Install CANN toolkit 9.1.0 + A3 ops package into the local docker container.
# Usage: bash install_cann.sh [--toolkit-only|--ops-only]
# Installers are read from C:\Users\wang\Downloads (WSL: /mnt/c/Users/wang/Downloads).
set -euo pipefail

C=${CANN_CONTAINER:-cann_container}
DL=/mnt/c/Users/wang/Downloads
TOOLKIT_RUN=Ascend-cann-toolkit_9.1.0_linux-x86_64.run
OPS_RUN=Ascend-cann-A3-ops_9.1.0_linux-x86_64.run
PKG_DIR=/root/cann_pkgs
ASCEND_HOME=/usr/local/Ascend

step() { echo -e "\n===== [install_cann] $* ====="; }

step "0. precheck"
docker start "$C" >/dev/null 2>&1 || true
# xz-utils needed by makeself extraction (openEuler: dnf)
docker exec "$C" bash -c 'command -v xz >/dev/null || dnf install -y xz || true'

step "1. copy installers into container"
docker exec "$C" mkdir -p "$PKG_DIR"
for f in "$TOOLKIT_RUN" "$OPS_RUN"; do
  if ! docker exec "$C" bash -c "[ -f $PKG_DIR/$f ]"; then
    echo "copying $f ..."
    docker cp "$DL/$f" "$C:$PKG_DIR/$f"
  else
    echo "$f already present, skip copy"
  fi
  docker exec "$C" chmod +x "$PKG_DIR/$f"
done

if [ "${1:-}" != "--ops-only" ]; then
  step "2. install toolkit (takes several minutes)"
  docker exec -i "$C" bash -c "cd $PKG_DIR && yes | ./$TOOLKIT_RUN --install"
fi

if [ "${1:-}" != "--toolkit-only" ]; then
  step "3. install A3 ops package (takes several minutes)"
  docker exec -i "$C" bash -c "cd $PKG_DIR && yes | ./$OPS_RUN --install"
fi

step "4. verify"
docker exec "$C" bash -c "
  ls $ASCEND_HOME/
  ls $ASCEND_HOME/ascend-toolkit/ 2>/dev/null
  source $ASCEND_HOME/ascend-toolkit/latest/set_env.sh 2>/dev/null || source $ASCEND_HOME/ascend-toolkit/latest/bin/setenv.bash 2>/dev/null || true
  which ccec cmake
  ccec --version 2>/dev/null | head -1 || true
"
echo "CANN install done."
