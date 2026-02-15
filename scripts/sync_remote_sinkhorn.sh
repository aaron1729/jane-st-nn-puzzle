#!/usr/bin/env bash
set -euo pipefail

# Periodically sync remote sinkhorn artifacts to local until done flag appears.

REMOTE_HOST="root@100.34.4.6"
REMOTE_PORT="50350"
REMOTE_DIR="/root/puzzle"
LOCAL_DIR="$(pwd)"
PREFIX="overnight"
INTERVAL=300
ONCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-dir) LOCAL_DIR="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --once) ONCE=1; shift 1 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

SSH_OPTS="-p ${REMOTE_PORT} -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12"
RSYNC_SSH="ssh ${SSH_OPTS}"

sync_once() {
  rsync -az -e "${RSYNC_SSH}" \
    --include "*/" \
    --include "${PREFIX}_*" \
    --include "sinkhorn_*.json" \
    --include "sinkhorn_*.log" \
    --exclude "*" \
    "${REMOTE_HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/"
}

check_done() {
  ssh ${SSH_OPTS} "${REMOTE_HOST}" "test -f '${REMOTE_DIR}/${PREFIX}_DONE.flag'"
}

sync_once
if (( ONCE == 1 )); then
  exit 0
fi

while true; do
  if check_done; then
    sync_once
    break
  fi
  sleep "${INTERVAL}"
  sync_once
done
