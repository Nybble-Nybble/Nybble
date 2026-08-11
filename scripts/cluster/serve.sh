#!/usr/bin/env bash
# Launch an OpenAI-compatible server with the model sharded across the cluster.
#
# Usage: serve.sh <hf-repo-or-path> [port]
#   MLX_PIPELINE=1  force pipeline parallelism instead of tensor parallelism.
#                   Only deepseek_v32 and ministral3 support this in mlx-lm 0.31.3;
#                   anything else raises "model does not support pipelining".
set -euo pipefail

MODEL="${1:?usage: serve.sh <hf-repo-or-path> [port]}"
PORT="${2:-8080}"
VENV="$HOME/.mlx-cluster/venv"
HOSTFILE="$HOME/.mlx-cluster/hosts.json"

[[ -f "$HOSTFILE" ]] || { echo "error: no $HOSTFILE - run hostfile.sh first" >&2; exit 1; }

# mlx.launch execs its target directly, so it must be an executable with a
# shebang. The venv console script qualifies; a bare .py file does not (it fails
# with "Permission denied" / exit 126).
TARGET="$VENV/bin/mlx_lm.server"
[[ -x "$TARGET" ]] || { echo "error: $TARGET missing - run bootstrap.sh" >&2; exit 1; }

ARGS=(--model "$MODEL" --host 0.0.0.0 --port "$PORT")
[[ "${MLX_PIPELINE:-0}" == "1" ]] && ARGS+=(--pipeline)

echo "==> serving $MODEL on :$PORT across:"
python3 -c 'import json,sys;[print("     ",h.get("ssh","?")) for h in json.load(open(sys.argv[1]))]' "$HOSTFILE"
echo

exec "$VENV/bin/mlx.launch" --hostfile "$HOSTFILE" --backend ring "$TARGET" "${ARGS[@]}"
