#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"

if [[ ! -f .env || ! -f .generated/compose.yaml ]]; then
  echo "not installed: run ./install.sh first" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

gpu_count=$(python3 - <<'PY'
import json
from pathlib import Path
print(len(json.loads(Path(".generated/gpu-info.json").read_text())))
PY
)
compose=(sudo docker compose --env-file .env -f .generated/compose.yaml)

echo "=== containers ==="
"${compose[@]}" ps

echo "=== GPUs ==="
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

echo "=== inference slots ==="
for ((index=0; index<gpu_count; index++)); do
  port=$((API_BASE_PORT + index))
  printf 'gpu=%d port=%d ' "${index}" "${port}"
  if ! curl -fsS --max-time 10 \
      -H "Authorization: Bearer ${API_KEY}" \
      "http://127.0.0.1:${port}/healthz" |
      jq -c '{ok,healthy_workers,workers:[.workers[]|{id,ok,running,pending,error}]}'; then
    echo '{"ok":false,"error":"health request failed"}'
  fi
done

echo "=== watchdog ==="
if [[ -f ${DATA_ROOT}/watchdog/status.json ]]; then
  jq '{ok,enabled,timestamp,instances}' "${DATA_ROOT}/watchdog/status.json"
else
  echo "watchdog has not written status yet"
fi

echo "=== registration ==="
if [[ -f ${DATA_ROOT}/reporter/status.json ]]; then
  jq . "${DATA_ROOT}/reporter/status.json"
else
  echo "reporter has not written status yet"
fi
