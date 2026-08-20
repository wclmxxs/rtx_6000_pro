#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"
mkdir -p .state

from_ami=false
case ${1:-} in
  "") ;;
  --from-ami) from_ami=true ;;
  -h|--help)
    echo "Usage: $0 [--from-ami]"
    echo "  --from-ami  Trust baked model sizes and reuse source-matched Docker images."
    exit 0
    ;;
  *)
    echo "unknown argument: $1" >&2
    echo "Usage: $0 [--from-ami]" >&2
    exit 2
    ;;
esac
if (( $# > 1 )); then
  echo "Usage: $0 [--from-ami]" >&2
  exit 2
fi

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

migrate_env_default() {
  local key=$1 old_value=$2 new_value=$3 current
  current=$(sed -n "s/^${key}=//p" .env | tail -1)
  if [[ ${current} == "${old_value}" ]]; then
    set_env "${key}" "${new_value}"
  fi
}

sudo -v
# This may be a cloned AMI whose old reporter was started by Docker before
# install.sh. Stop it immediately, and once more after bootstrap_host.sh in
# case restarting Docker brought it back through restart: unless-stopped.
sudo docker stop minimax-h3-reporter >/dev/null 2>&1 || true
sudo docker stop minimax-h3-watchdog >/dev/null 2>&1 || true
if [[ ${from_ami} == true ]]; then
  echo "AMI fast path: validating baked host runtime"
  for command in curl jq openssl python3 flock nvidia-smi docker; do
    command -v "${command}" >/dev/null 2>&1 || {
      echo "AMI fast path requires ${command}; run ./install.sh once to repair the host" >&2
      exit 1
    }
  done
  nvidia-smi -L >/dev/null
  sudo systemctl enable --now docker >/dev/null
  sudo docker info >/dev/null
  sudo docker compose version
else
  sudo scripts/bootstrap_host.sh
fi
exec 9>.state/install.lock
if ! flock -n 9; then
  echo "another install.sh process is running" >&2
  exit 1
fi

# Keep the old reporter stopped while refreshing the node identity.
sudo docker stop minimax-h3-reporter >/dev/null 2>&1 || true
sudo docker stop minimax-h3-watchdog >/dev/null 2>&1 || true

if [[ -z $(sed -n 's/^API_KEY=//p' .env) ]]; then
  set_env API_KEY "$(openssl rand -hex 32)"
fi
if [[ -z $(sed -n 's/^WATCHDOG_IMAGE=//p' .env) ]]; then
  set_env WATCHDOG_IMAGE "minimax-h3-watchdog:20260821-v1"
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

configured_advertise_host=$(sed -n 's/^ADVERTISE_HOST=//p' .env)
detected_advertise_host=$(detect_imds public-ipv4 || true)
if [[ -n ${detected_advertise_host} ]]; then
  advertise_host=${detected_advertise_host}
  if [[ ${configured_advertise_host} != "${advertise_host}" ]]; then
    echo "Refreshing ADVERTISE_HOST from AWS IMDS: ${configured_advertise_host:-<empty>} -> ${advertise_host}"
  fi
  set_env ADVERTISE_HOST "${advertise_host}"
else
  advertise_host=${configured_advertise_host}
  [[ -n ${advertise_host} ]] || {
    echo "AWS IMDS did not return public-ipv4; set ADVERTISE_HOST to this node's public IPv4" >&2
    exit 1
  }
fi
if ! require_public_ipv4 "${advertise_host}"; then
  echo "ADVERTISE_HOST must be a public IPv4 address; got ${advertise_host}" >&2
  exit 1
fi

configured_instance_id=$(sed -n 's/^INSTANCE_ID=//p' .env)
detected_instance_id=$(detect_imds instance-id || true)
if [[ -n ${detected_instance_id} ]]; then
  instance_id=${detected_instance_id}
  if [[ ${configured_instance_id} != "${instance_id}" ]]; then
    echo "Refreshing INSTANCE_ID from AWS IMDS: ${configured_instance_id:-<empty>} -> ${instance_id}"
  fi
  set_env INSTANCE_ID "${instance_id}"
else
  instance_id=${configured_instance_id:-$(hostname)}
  [[ -n ${instance_id} ]] || { echo "unable to determine INSTANCE_ID" >&2; exit 1; }
  set_env INSTANCE_ID "${instance_id}"
fi

set_env HOST_UID "$(id -u)"
set_env HOST_GID "$(id -g)"

# Preserve operator overrides, but move installations that still carry the
# previous conservative defaults to the validated long-video SM120 profile.
migrate_env_default RELEASE_ID \
  h3-rtx6000pro-20260819-v1 h3-rtx6000pro-20260820-v2
migrate_env_default RELEASE_ID \
  h3-rtx6000pro-20260820-v2 h3-rtx6000pro-20260820-v3
migrate_env_default RELEASE_ID \
  h3-rtx6000pro-20260820-v3 h3-rtx6000pro-20260820-v4
migrate_env_default RELEASE_ID \
  h3-rtx6000pro-20260820-v4 h3-rtx6000pro-20260821-v5
migrate_env_default CACHE_DIT_WARMUP_STEPS 2 1
migrate_env_default CACHE_DIT_RESIDUAL_DIFF_THRESHOLD 0.35 0.24
migrate_env_default SOL_ATTN_TAU_START 1.2 1.5
migrate_env_default SOL_ATTN_TAU_END 0.8 1.5
migrate_env_default SOL_ATTN_STRICT false true
migrate_env_default SOL_ATTN_DENSE_PERCENT 0.2 0
migrate_env_default SOL_ATTN_INT8_QK false true
migrate_env_default SOL_ATTN_DENSE_BLOCKS 0-2,-1 ""

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

sudo mkdir -p \
  "${DATA_ROOT}/models" \
  "${DATA_ROOT}/reporter" \
  "${DATA_ROOT}/watchdog" \
  "${DATA_ROOT}/warmup"
for ((index=0; index<gpu_count; index++)); do
  sudo mkdir -p \
    "${DATA_ROOT}/slots/${index}/input" \
    "${DATA_ROOT}/slots/${index}/output" \
    "${DATA_ROOT}/slots/${index}/temp" \
    "${DATA_ROOT}/slots/${index}/user" \
    "${DATA_ROOT}/slots/${index}/api-data"
  # install owns the maintenance window; do not carry a quarantine marker
  # from the source AMI or an interrupted previous recovery into warmup.
  sudo rm -f "${DATA_ROOT}/slots/${index}/api-data/watchdog.json"
done
sudo chown -R "$(id -u):$(id -g)" "${DATA_ROOT}"

mkdir -p .state .generated
if [[ ! -x .state/model-venv/bin/python ]]; then
  python3 -m venv .state/model-venv
  .state/model-venv/bin/pip install --upgrade pip
  .state/model-venv/bin/pip install 'huggingface_hub>=0.34,<2' 'hf_xet>=1.1,<2'
fi

model_args=(--root "${DATA_ROOT}/models")
if [[ ${from_ami} == true ]]; then
  model_args+=(--trust-existing-size --prefetch-existing)
fi
HF_TOKEN=${HF_TOKEN:-} .state/model-venv/bin/python scripts/download_models.py \
  "${model_args[@]}"

docker_cmd=(sudo docker)
image_source_label=com.capcut.minimax_h3.source_digest
build_image() {
  local dockerfile=$1 image=$2 source_digest existing_digest
  shift 2
  source_digest=$(python3 scripts/source_digest.py "${dockerfile}" "$@")
  existing_digest=$(
    "${docker_cmd[@]}" image inspect \
      --format "{{ index .Config.Labels \"${image_source_label}\" }}" \
      "${image}" 2>/dev/null || true
  )
  if [[ ${from_ami} == true && ${existing_digest} == "${source_digest}" ]]; then
    echo "AMI fast path: reusing source-matched Docker image ${image}"
    return
  fi
  if [[ ${from_ami} == true ]]; then
    if [[ -n ${existing_digest} ]]; then
      echo "AMI fast path: source changed; rebuilding Docker image ${image}"
    else
      echo "AMI fast path: image is missing a source digest; rebuilding ${image}"
    fi
  fi
  "${docker_cmd[@]}" build --progress=plain \
    --label "${image_source_label}=${source_digest}" \
    -f "${dockerfile}" -t "${image}" .
}
build_image docker/Dockerfile.worker "${WORKER_IMAGE}"
build_image docker/Dockerfile.api "${API_IMAGE}" api
build_image docker/Dockerfile.reporter "${REPORTER_IMAGE}" reporter
build_image docker/Dockerfile.watchdog "${WATCHDOG_IMAGE}" watchdog

python3 scripts/generate_compose.py \
  --data-root "${DATA_ROOT}" \
  --advertise-host "${ADVERTISE_HOST}" \
  --instance-id "${INSTANCE_ID}" \
  --base-port "${API_BASE_PORT}" \
  --release-id "${RELEASE_ID}" \
  --worker-image "${WORKER_IMAGE}" \
  --api-image "${API_IMAGE}" \
  --watchdog-image "${WATCHDOG_IMAGE}"

compose=(sudo docker compose --env-file .env -f .generated/compose.yaml)
services=()
for ((index=0; index<gpu_count; index++)); do
  services+=("h3-comfy-${index}" "h3-api-${index}")
done
"${compose[@]}" stop h3-reporter h3-watchdog >/dev/null 2>&1 || true
compose_up_args=(-d --remove-orphans)
if [[ ${from_ami} == true ]]; then
  # EC2 AMIs preserve Docker container metadata, including stale health state
  # and device bindings from the source machine. Recreate containers while
  # retaining all bind-mounted models and data directories.
  compose_up_args+=(--force-recreate)
fi
"${compose[@]}" up "${compose_up_args[@]}" "${services[@]}"

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

warmup_args=(
  --gpu-count "${gpu_count}"
  --base-port "${API_BASE_PORT}"
  --parallelism "${WARMUP_PARALLELISM}"
  --release-id "${RELEASE_ID}"
  --marker-root "${DATA_ROOT}/warmup"
)
if [[ ${from_ami} == true ]]; then
  # AMIs preserve old warmup markers but not GPU memory. Force warmup on the
  # cloned machine so registration never exposes an entirely cold worker.
  warmup_args+=(--force)
fi
python3 scripts/warmup.py "${warmup_args[@]}"

watchdog_started_at=$(date +%s)
"${compose[@]}" up -d h3-watchdog

deadline=$((SECONDS + 60))
watchdog_ready=false
while (( SECONDS < deadline )); do
  if [[ -f ${DATA_ROOT}/watchdog/status.json ]] \
    && jq -e --argjson started "${watchdog_started_at}" \
      '.ok == true and .timestamp >= $started' \
      "${DATA_ROOT}/watchdog/status.json" >/dev/null 2>&1; then
    watchdog_ready=true
    break
  fi
  sleep 2
done
if [[ ${watchdog_ready} != true ]]; then
  "${compose[@]}" logs --tail 100 h3-watchdog
  echo "services are warm, but watchdog did not become ready" >&2
  exit 1
fi

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
