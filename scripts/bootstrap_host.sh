#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "bootstrap_host.sh must run as root" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver is required before installation (nvidia-smi not found)" >&2
  exit 1
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "NVIDIA driver is installed but no usable GPU was found" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git git-lfs jq openssl python3 python3-venv gnupg util-linux

if ! command -v docker >/dev/null 2>&1; then
  apt-get install -y --no-install-recommends docker.io
fi

if ! docker compose version >/dev/null 2>&1; then
  if apt-cache show docker-compose-v2 >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends docker-compose-v2
  elif apt-cache show docker-compose-plugin >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends docker-compose-plugin
  else
    echo "Docker Compose v2 package is unavailable" >&2
    exit 1
  fi
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL \
    https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y --no-install-recommends nvidia-container-toolkit
fi

nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker
docker info >/dev/null
docker compose version
nvidia-smi -L
