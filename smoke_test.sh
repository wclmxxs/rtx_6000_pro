#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"

set -a
# shellcheck disable=SC1091
source .env
set +a

base_url="http://127.0.0.1:${API_BASE_PORT}"
task_id=$(
  curl -fsS -X POST "${base_url}/ic/capcut/edit_gateway/v2/video_generation" \
    -H 'Content-Type: application/json' \
    -d '{
      "model":"MiniMax-H3",
      "content":[{"type":"text","text":"A cinematic sunrise over a quiet lake, static camera."}],
      "resolution":"704P",
      "duration":4,
      "ratio":"16:9",
      "num_inference_steps":8
    }' | jq -er .task_id
)
echo "task_id=${task_id}"

while true; do
  result=$(
    curl -fsS -X POST \
      "${base_url}/ic/capcut/edit_gateway/v2/query/video_generation" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"MiniMax-H3\",\"task_id\":\"${task_id}\"}"
  )
  status=$(jq -r .task.status <<<"${result}")
  echo "$(date '+%H:%M:%S') ${status}"
  case "${status}" in
    succeeded) jq . <<<"${result}"; break ;;
    failed) jq . <<<"${result}"; exit 1 ;;
  esac
  sleep 2
done

