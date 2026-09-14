#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
set -e

while [[ $# -gt 0 ]]; do
    case $1 in
    -l | --linker)
        linker=$2
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

for arg in "$@"
do
    # HC_MERGE_GUARD: ld.lld -m aicorelinux merges .text.<mangle> sections in
    # place and cannot re-read an already-merged object ("unknown file type").
    # merge_aiv/aic_obj_text are always-run custom targets, so a rebuild with
    # no source changes would re-merge merged objects and fail. Skip objects
    # that no longer carry any .text.<mangle> section (nothing to merge).
    need_merge=1
    od=""
    if command -v llvm-objdump >/dev/null 2>&1; then
        od=llvm-objdump
    elif command -v objdump >/dev/null 2>&1; then
        od=objdump
    fi
    if [ -n "$od" ]; then
        secs=$("$od" --section-headers "${arg}" 2>/dev/null)
        if [ $? -eq 0 ] && ! echo "${secs}" | grep -q '\.text\.'; then
            need_merge=0
        fi
    fi
    if [ "$need_merge" -eq 1 ]; then
        echo -n "${linker} -m aicorelinux -Ttext=0 ${arg} -o ${arg}"
        # template kernel text sections will be named as .text.<mangle> like .text._Z11hello_worldILi300EhEvT0_f
        # to remain consistent with none template kernels, merge them into .text
        ${linker} -m aicorelinux -Ttext=0 ${arg} -o ${arg}
    else
        echo -n "(HC_MERGE_GUARD: already merged, skip) ${arg}"
    fi
done
