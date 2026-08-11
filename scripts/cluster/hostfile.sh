#!/usr/bin/env bash
# Generate ~/.mlx-cluster/hosts.json for mlx.launch.
#
# Prefers Thunderbolt. Falls back to ethernet if no TB bridge is configured, and
# tells you so loudly, because on Wi-Fi tensor parallelism is unusable.
#
# Usage: hostfile.sh <host1> <host2> [host3 ...]
set -euo pipefail

OUT="$HOME/.mlx-cluster/hosts.json"
VENV="$HOME/.mlx-cluster/venv"
[[ $# -ge 2 ]] || { echo "usage: hostfile.sh <host1> <host2> [...]" >&2; exit 1; }

HOSTS=$(IFS=,; echo "$*")
mkdir -p "$(dirname "$OUT")"

echo "==> probing $HOSTS over thunderbolt"
if "$VENV/bin/mlx.distributed_config" --hosts "$HOSTS" --over thunderbolt \
     --backend ring --auto-setup --output-hostfile "$OUT" 2>/dev/null; then
  echo "==> thunderbolt ring configured"
else
  echo
  echo "!! no Thunderbolt topology found. Falling back to ethernet."
  echo "!! If these nodes are on Wi-Fi, tensor parallelism will crawl."
  echo "!! Check with: scripts/cluster/netcheck.sh <peer>"
  echo
  "$VENV/bin/mlx.distributed_config" --hosts "$HOSTS" --over ethernet \
    --backend ring --output-hostfile "$OUT"
fi

echo
echo "wrote $OUT"
cat "$OUT"
