#!/usr/bin/env bash
# Prepare one Mac to be a node in the MLX inference cluster.
# Run this identically on every node. Safe to re-run.
set -euo pipefail

VENV="$HOME/.mlx-cluster/venv"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "error: needs Apple Silicon macOS" >&2
  exit 1
fi

# The venv path must be byte-identical across nodes: mlx.launch SSHes in and
# execs the same absolute path on every machine.
if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> creating $VENV"
mkdir -p "$(dirname "$VENV")"
uv venv --python 3.12 "$VENV" >/dev/null
uv pip install --python "$VENV/bin/python" --upgrade mlx-lm

# macOS caps GPU-wired memory at ~75% of RAM, which is wasteful on a dedicated
# node. Raise it to ~80%, leaving headroom for the OS.
TOTAL_MB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 ))
TARGET_MB=$(( TOTAL_MB * 80 / 100 ))
echo "==> raising GPU wired limit to ${TARGET_MB} MB of ${TOTAL_MB} MB"
sudo sysctl -w iogpu.wired_limit_mb="$TARGET_MB"

# Not persistent across reboot. To make it stick:
#   echo "iogpu.wired_limit_mb=$TARGET_MB" | sudo tee -a /etc/sysctl.conf

echo
echo "node ready"
"$VENV/bin/python" -c 'import mlx.core as mx, mlx_lm; print(f"  mlx {mx.__version__}  mlx-lm {mlx_lm.__version__}")'
echo "  hostname: $(scutil --get LocalHostName).local"
echo
echo "next: enable Remote Login (System Settings > General > Sharing) and"
echo "      ssh-copy-id between every pair of nodes."
