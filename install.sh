#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"
mkdir -p .state

if [[ ! -f .env ]]; then
  cp config/env.example .env
fi
chmod 600 .env

set_env() {
  local key=$1 value=$2
  if grep -q "^${key}=" .env; then
    sed -i "s|^${key}=.*$|${key}=${value}|" .env
  else
    printf '%s=%s\n' "${key}" "${value}" >> .env
  fi
}

sudo -v
sudo scripts/bootstrap_host.sh
exec 9>.state/install.lock
if ! flock -n 9; then
  echo "another install.sh process is running" >&2
  exit 1
fi

if [[ -z $(sed -n 's/^API_KEY=//p' .env) ]]; then
  set_env API_KEY "$(openssl rand -hex 32)"
fi

detect_imds() {
  local path=$1 token
  token=$(curl -fsS --connect-timeout 1 -X PUT \
    http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)
  [[ -n ${token} ]] || return 1
  curl -fsS --connect-timeout 1 \
    -H "X-aws-ec2-metadata-token: ${token}" \
    "http://169.254.169.254/latest/meta-data/${path}" 2>/dev/null
}

require_public_ipv4() {
  python3 - "$1" <<'PY'
import ipaddress
import sys

try:
    address = ipaddress.ip_address(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if address.version == 4 and address.is_global else 1)
PY
}

advertise_host=$(sed -n 's/^ADVERTISE_HOST=//p' .env)
if [[ -z ${advertise_host} ]]; then
  advertise_host=$(detect_imds public-ipv4 || true)
  [[ -n ${advertise_host} ]] || {
    echo "AWS IMDS did not return public-ipv4; set ADVERTISE_HOST to this node's public IPv4" >&2
    exit 1
  }
  set_env ADVERTISE_HOST "${advertise_host}"
fi
if ! require_public_ipv4 "${advertise_host}"; then
  echo "ADVERTISE_HOST must be a public IPv4 address; got ${advertise_host}" >&2
  exit 1
fi

instance_id=$(sed -n 's/^INSTANCE_ID=//p' .env)
if [[ -z ${instance_id} ]]; then
  instance_id=$(detect_imds instance-id || hostname)
  [[ -n ${instance_id} ]] || { echo "unable to determine INSTANCE_ID" >&2; exit 1; }
  set_env INSTANCE_ID "${instance_id}"
fi

set_env HOST_UID "$(id -u)"
set_env HOST_GID "$(id -g)"

set -a
# shellcheck disable=SC1091
source .env
set +a

gpu_count=$(nvidia-smi -L | sed -n 's/^GPU [0-9][0-9]*:.*/x/p' | wc -l | tr -d ' ')
if (( gpu_count < 1 )); then
  echo "no GPUs detected" >&2
  exit 1
fi

echo "Detected ${gpu_count} GPUs on ${INSTANCE_ID} (${ADVERTISE_HOST})"

sudo mkdir -p "${DATA_ROOT}/models" "${DATA_ROOT}/reporter" "${DATA_ROOT}/warmup"
for ((index=0; index<gpu_count; index++)); do
  sudo mkdir -p \
    "${DATA_ROOT}/slots/${index}/input" \
    "${DATA_ROOT}/slots/${index}/output" \
    "${DATA_ROOT}/slots/${index}/temp" \
    "${DATA_ROOT}/slots/${index}/user" \
    "${DATA_ROOT}/slots/${index}/api-data"
done
sudo chown -R "$(id -u):$(id -g)" "${DATA_ROOT}"

mkdir -p .state .generated
if [[ ! -x .state/model-venv/bin/python ]]; then
  python3 -m venv .state/model-venv
  .state/model-venv/bin/pip install --upgrade pip
  .state/model-venv/bin/pip install 'huggingface_hub>=0.34,<2' 'hf_xet>=1.1,<2'
fi

HF_TOKEN=${HF_TOKEN:-} .state/model-venv/bin/python scripts/download_models.py \
  --root "${DATA_ROOT}/models"

docker_cmd=(sudo docker)
"${docker_cmd[@]}" build --progress=plain \
  -f docker/Dockerfile.worker -t "${WORKER_IMAGE}" .
"${docker_cmd[@]}" build --progress=plain \
  -f docker/Dockerfile.api -t "${API_IMAGE}" .
"${docker_cmd[@]}" build --progress=plain \
  -f docker/Dockerfile.reporter -t "${REPORTER_IMAGE}" .

python3 scripts/generate_compose.py \
  --data-root "${DATA_ROOT}" \
  --advertise-host "${ADVERTISE_HOST}" \
  --instance-id "${INSTANCE_ID}" \
  --base-port "${API_BASE_PORT}" \
  --release-id "${RELEASE_ID}" \
  --worker-image "${WORKER_IMAGE}" \
  --api-image "${API_IMAGE}"

compose=(sudo docker compose --env-file .env -f .generated/compose.yaml)
services=()
for ((index=0; index<gpu_count; index++)); do
  services+=("h3-comfy-${index}" "h3-api-${index}")
done
"${compose[@]}" stop h3-reporter >/dev/null 2>&1 || true
"${compose[@]}" up -d --remove-orphans "${services[@]}"

deadline=$((SECONDS + 900))
while (( SECONDS < deadline )); do
  healthy=0
  for ((index=0; index<gpu_count; index++)); do
    port=$((API_BASE_PORT + index))
    if curl -fsS "http://127.0.0.1:${port}/healthz" \
      -H "Authorization: Bearer ${API_KEY}" >/dev/null 2>&1; then
      healthy=$((healthy + 1))
    fi
  done
  if (( healthy == gpu_count )); then
    break
  fi
  sleep 5
done
if (( healthy != gpu_count )); then
  "${compose[@]}" logs --tail 200
  echo "only ${healthy}/${gpu_count} API instances became healthy" >&2
  exit 1
fi

python3 scripts/warmup.py \
  --gpu-count "${gpu_count}" \
  --base-port "${API_BASE_PORT}" \
  --parallelism "${WARMUP_PARALLELISM}" \
  --release-id "${RELEASE_ID}" \
  --marker-root "${DATA_ROOT}/warmup"

report_started_at=$(date +%s)
"${compose[@]}" up -d h3-reporter

deadline=$((SECONDS + 60))
catalog_success=false
while (( SECONDS < deadline )); do
  if [[ -f ${DATA_ROOT}/reporter/status.json ]] \
    && jq -e --argjson started "${report_started_at}" \
      '.catalog_success == true
       and .timestamp >= $started
       and .healthy_instances == .instance_count' \
      "${DATA_ROOT}/reporter/status.json" >/dev/null 2>&1; then
    catalog_success=true
    break
  fi
  sleep 2
done
if [[ ${catalog_success} != true ]]; then
  "${compose[@]}" logs --tail 100 h3-reporter
  echo "services are running, but ReportCatalog registration did not succeed" >&2
  exit 1
fi

echo "READY: ${gpu_count} GPU instances"
echo "PSM=${PSM}"
echo "SERVICE_ID=${SERVICE_ID}"
echo "Ports: ${API_BASE_PORT}-$((API_BASE_PORT + gpu_count - 1))"
jq '{catalog_success,healthy_instances,instance_count,catalog_response}' \
  "${DATA_ROOT}/reporter/status.json"
