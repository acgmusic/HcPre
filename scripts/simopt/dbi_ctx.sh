#!/bin/bash
docker exec cann_container bash -c '
echo "== log 行数: $(wc -l < /tmp/dbi_run.log)"
grep -vE "DRVSTUB" /tmp/dbi_run.log | tail -20
'
