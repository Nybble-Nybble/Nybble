#!/usr/bin/env bash
# Measure the interconnect to a peer and report the tokens/sec ceiling it implies.
#
# Tensor parallelism costs 2 blocking all-reduces per transformer layer per token
# (mlx_lm/models/qwen3.py shard() -> sharded-to-all -> mx.distributed.all_sum).
# So the network alone caps you at:  1 / (2 * layers * RTT)
#
# Usage: netcheck.sh <peer-host> [layers]
set -euo pipefail

PEER="${1:?usage: netcheck.sh <peer-host> [layers]}"
LAYERS="${2:-64}"   # Qwen3-32B

echo "==> pinging $PEER (assuming $LAYERS layers)"
# macOS: "round-trip min/avg/max/stddev = 0.1/0.2/0.3/0.0 ms"
STATS=$(ping -c 20 -i 0.2 "$PEER" 2>/dev/null | awk '/min\/avg\/max/ {print $4}')
MIN=$(cut -d/ -f1 <<<"$STATS")
RTT=$(cut -d/ -f2 <<<"$STATS")

if [[ -z "$RTT" ]]; then
  echo "error: no reply from $PEER" >&2
  exit 1
fi

# Which interface are we actually using? Thunderbolt bridge is bridge0 / 169.254.x.
ROUTE=$(route -n get "$PEER" 2>/dev/null | awk '/interface:/ {print $2}')
IFACE_NAME=$(networksetup -listallhardwareports 2>/dev/null \
  | awk -v d="$ROUTE" '/^Hardware Port:/{p=substr($0,16)} /^Device: /{if ($2==d) print p}')

TP=$(awk -v r="$RTT" -v l="$LAYERS" 'BEGIN{printf "%.1f", 1000/(2*l*r)}')
PP=$(awk -v r="$RTT" 'BEGIN{printf "%.1f", 1000/r}')

echo
echo "  RTT             ${MIN} ms min / ${RTT} ms avg"
echo "  interface       ${ROUTE:-?} ${IFACE_NAME:+($IFACE_NAME)}"
echo "  tensor-parallel ceiling   ${TP} tok/s"
echo "  pipeline-parallel ceiling ${PP} tok/s"
echo
# ICMP RTT overstates MLX's cost a little (ring keeps persistent TCP sockets and
# skips the userspace ICMP path), so treat this as a screening test, not a
# benchmark. It is decisive at the order-of-magnitude level, which is all that
# matters for the shard-or-not decision.

VERDICT=$(awk -v t="$TP" 'BEGIN{ if (t>=30) print "good"; else if (t>=10) print "usable"; else print "bad" }')
case "$VERDICT" in
  good)   echo "  OK - shard away." ;;
  usable) echo "  Marginal. Works, but you are leaving speed on the table." ;;
  bad)    echo "  DO NOT SHARD over this link. You are almost certainly on Wi-Fi or WAN."
          echo "  Use a Thunderbolt cable, or run this peer as an independent endpoint."
          exit 2 ;;
esac
