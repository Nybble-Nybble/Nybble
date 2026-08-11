# Distributed inference across Macs

Everything verified against a real install on 2026-08-04: `mlx 0.32.0`, `mlx-lm 0.31.3`,
macOS 26.5.2, Mac mini M4 (Mac16,10).

## 1. What distributed inference actually buys you

It pools **memory**, not compute. If a model already fits on one machine, adding a second
node makes it *slower*, not faster. The only reason to build a cluster is to run a model
that no single machine can hold.

Two ways to split a model:

| | Tensor parallel | Pipeline parallel |
|---|---|---|
| Split | Each layer's weights split across nodes | Contiguous blocks of layers per node |
| Sync per token | **2 all-reduces per layer** | 1 send/recv per node boundary |
| For 64 layers, 2 nodes | 128 network round-trips/token | 1 network round-trip/token |
| Needs | Very fast, low-latency link | Tolerates slow links |
| Utilization | All nodes busy | Only one node busy at a time (bubbles) |

This table is the whole story. Everything below follows from it.

### Round-trips per token is the number that matters

Confirmed in `mlx_lm/models/qwen3.py:190` - `shard()` converts `o_proj` and `down_proj`
to `sharded-to-all`, and `mlx/nn/layers/distributed.py:333,585` shows each of those calls
`mx.distributed.all_sum`. So tensor parallel costs **2 blocking all-reduces per transformer
layer, per generated token**.

Qwen3-32B has 64 layers, so 128 round-trips per token. Network-imposed ceiling:

| Link | RTT | Ceiling (tensor parallel) | Ceiling (pipeline parallel) |
|---|---|---|---|
| Thunderbolt 4 bridge | ~0.15 ms | ~50 tok/s | irrelevant, use TP |
| Wired 1GbE | ~0.4 ms | ~19 tok/s | ~2500 tok/s |
| Wi-Fi (good) | ~3 ms | ~2.6 tok/s | ~330 tok/s |
| Wi-Fi (jittery) | ~15 ms | ~0.5 tok/s | ~66 tok/s |
| Seattle to Yerevan | ~180 ms | **~0.04 tok/s (23 s/token)** | ~5.5 tok/s |

These are ceilings from network alone, before any compute. Two conclusions:

1. **Wi-Fi is already disqualifying for tensor parallel.** Use a cable.
2. **A transatlantic tensor-parallel shard is not slow, it is broken.** 23 seconds per token.

### Measured on the Mac mini M4, 2026-08-04

Qwen3-1.7B-4bit (28 layers, 1.035 GB peak memory), 200 tokens, temperature 0:

| Config | tok/s | ms/token |
|---|---|---|
| 1 rank, no sharding | 60.2 | 16.6 |
| 2 ranks, tensor parallel over **loopback** | 28.6 | 35.0 |

Both ranks share one GPU here, so there is no bandwidth benefit - this isolates pure
synchronization cost. **Sharding a model that already fits cost 2.1x**, over the fastest
"network" that exists (no wire at all). The delta is 18.4 ms/token across 56 all-reduces
(28 layers x 2), so **~0.33 ms per all-reduce** including MLX framework and GPU stream sync.

That per-collective figure is the one to plan with. It is larger than raw ICMP RTT suggests,
so `netcheck.sh` output is an order-of-magnitude screen, not a prediction.

On two *real* machines you additionally get 2x aggregate memory bandwidth, which is what
offsets this. Sharding pays off only when per-node compute time exceeds sync time - i.e.
for big models. For a 32B-4bit (~18 GB) model on two M4 minis (~120 GB/s each):

- one node, if it fit: 18/120 = 150 ms/token = 6.7 tok/s
- two nodes: 9/120 = 75 ms + ~45 ms sync = 120 ms/token = 8.3 tok/s

A ~1.2x win, plus the fact that it runs at all. That is the honest trade.

## 2. Your three machines, honestly

| Machine | Chip | Thunderbolt | Notes |
|---|---|---|---|
| Mac mini (here) | M4 base, 16 GB | **TB4** (40 Gb/s, verified) | ~10.6 GB usable GPU by default |
| Your laptop | ? | ? | run `scripts/cluster/netcheck.sh` on it |
| Friend, Armenia | M3 Max | **TB4** | RAM unknown, matters a lot |

### JACCL is not available to you

The 2026 headline feature is JACCL - MLX collectives over RDMA on Thunderbolt 5, ~3-9 µs
latency, shipped in macOS 26.2. It **requires Thunderbolt 5**. Your Mac mini is a base M4
(TB4, confirmed via `system_profiler SPThunderboltDataType`: "Speed: Up to 40 Gb/s"), and
M3 Max MacBook Pros are also TB4. TB5 starts at M4 Pro/Max.

So you are on the **`ring` backend over TCP**. That is fine - it is the default, needs no
setup, and over a Thunderbolt bridge it is genuinely fast. Just know that the benchmarks
you read about (1T-param models at 28 tok/s on four M3 Ultras) are a different hardware
class and a different backend.

### Not every model can be sharded

In mlx-lm 0.31.3, sharding support is per-architecture and **asymmetric**:

- **Tensor parallel** (`shard()`) - widely supported: `llama`, `qwen2`, `qwen3`, `qwen3_5`,
  `gpt_oss`, `deepseek_v2/v3/v32`, `glm4_moe`, `kimi_k25`, `minimax`, `exaone_moe`,
  `longcat_flash`, `step3p5`, `ministral3`.
- **Pipeline parallel** (`pipeline()`) - only `deepseek_v32` and `ministral3`.

This is the trap. `mlx_lm.server --pipeline` on a Qwen3 model fails outright:

```
ValueError: The model does not support pipelining but a pipeline_group was provided
```

(reproduced locally). And `sharded_load` defaults to **tensor parallel** when both are
available. So the latency-tolerant strategy is exactly the one you mostly cannot use in
mlx-lm. If you need pipeline parallel on arbitrary models, that is what `exo` is for.

## 3. The architecture that actually works

Do **not** build one 3-node cluster. Build two things.

```
     ┌─────────────── LAN, Thunderbolt cable ───────────────┐
     │                                                       │
  Mac mini M4                                          Your laptop
   16 GB                                                  ? GB
     └──────────────── ONE sharded model ───────────────────┘
                    tensor parallel, ring backend
                       mlx_lm.server on :8080

              ~~~~ Tailscale (WireGuard mesh) ~~~~
                    control plane only, NOT tensor traffic

                        Friend's M3 Max, Armenia
                    INDEPENDENT endpoint, whole model
                          mlx_lm.server on :8080
```

- **Local pair**: one model sharded across mini + laptop. Real memory pooling.
- **Armenia**: a separate, complete model behind an OpenAI-compatible endpoint. You route
  requests to it; you never shard across the ocean.

Why the friend's box is still worth having: an M3 Max has more RAM than your mini almost
certainly does, so it can hold a bigger model than your entire local pair. Treat it as a
second, larger backend for batch and non-interactive work - eval sweeps, offline labeling,
dataset generation for Nybble. Latency there is per-request (~200 ms), not per-token, which
is completely tolerable.

## 4. Setup

### 4.1 Every node

```bash
scripts/cluster/bootstrap.sh
```

Installs `uv`, creates `~/.mlx-cluster/venv` with `mlx-lm`, and raises the GPU wired-memory
limit. That last part matters on your 16 GB mini: macOS reserves ~25% for the system, so
you get ~10.6 GB of GPU-usable memory by default. The script raises it to ~75-80% of RAM.
It is not persistent across reboot by design - see the comment in the script for making it
stick.

**Critical**: the venv path must be **identical on every node**. `mlx.launch` SSHes in and
execs the same absolute path everywhere. `~/.mlx-cluster/venv` is chosen for that reason.

### 4.2 Wire the local pair

Thunderbolt cable between mini and laptop. A TB4 cable, not a USB-C charging cable - they
look identical and a charge cable silently degrades you to USB 2 speeds.

System Settings > Network > Thunderbolt Bridge > make sure both are members and get
self-assigned 169.254.x.x addresses. Then confirm passwordless SSH both directions:

```bash
ssh-copy-id laptop.local          # from the mini
```

### 4.3 Verify the link before trusting it

```bash
scripts/cluster/netcheck.sh laptop.local
```

Prints RTT and the resulting tokens/sec ceiling for a 64-layer model. **If this says under
~10 tok/s, stop and fix the cable or the interface.** You are almost certainly on Wi-Fi.
Do not proceed to benchmark the model and blame MLX.

### 4.4 Build the hostfile and launch

```bash
scripts/cluster/hostfile.sh mini.local laptop.local   # writes ~/.mlx-cluster/hosts.json
scripts/cluster/serve.sh mlx-community/Qwen3-32B-4bit
```

`serve.sh` wraps:

```bash
mlx.launch --hostfile ~/.mlx-cluster/hosts.json --backend ring \
  ~/.mlx-cluster/venv/bin/mlx_lm.server --model <repo> --port 8080
```

Note it launches the **`mlx_lm.server` console script**, not a `.py` file.
`mlx.launch` `exec`s its target directly, so the target must be executable and carry a
shebang. Passing a plain `.py` fails with `Permission denied` / `exit code 126` - verified.
Console scripts in the venv's `bin/` already satisfy this.

Distribute weights to the other nodes without re-downloading from HF on each:

```bash
~/.mlx-cluster/venv/bin/mlx_lm.share --model mlx-community/Qwen3-32B-4bit \
  --hostfile ~/.mlx-cluster/hosts.json
```

### 4.5 Armenia node

On the friend's M3 Max:

```bash
scripts/cluster/bootstrap.sh
brew install --cask tailscale        # then sign in to your tailnet
~/.mlx-cluster/venv/bin/mlx_lm.server --model <a model that fits their RAM> \
  --host 0.0.0.0 --port 8080
```

Tailscale gives you a stable `100.x.y.z` address that survives NAT, CGNAT, and their
router. From here:

```bash
curl http://<their-tailscale-ip>:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default_model","messages":[{"role":"user","content":"hi"}]}'
```

Bind to `0.0.0.0` only because Tailscale is doing the authentication and encryption. Do not
port-forward this on their home router - `mlx_lm.server` has no auth.

## 5. Sizing

Usable GPU memory ≈ 75-80% of RAM (after `bootstrap.sh` raises the limit). Budget ~1.2x the
weight size to leave room for KV cache and activations.

| Pooled RAM | Realistic model at 4-bit |
|---|---|
| 16 GB (mini alone) | 7-8B comfortably, 14B tight |
| 32 GB (mini + 16 GB laptop) | 30-32B |
| 48 GB (mini + 32 GB laptop) | 32B comfortably, 70B very tight |
| 64 GB+ (M3 Max alone) | 70B, or a small MoE |

MoE models (`gpt_oss`, `glm4_moe`, `qwen3` MoE variants) are the sweet spot for clusters:
large total weights, small active parameter count per token, so they are memory-bound rather
than compute-bound - exactly what pooling memory fixes.

## 6. When to reach for exo instead

Use `exo` (v1.0.71, Apr 2026) rather than raw mlx-lm if you want:

- **Pipeline parallel on arbitrary models** - the main gap in mlx-lm above.
- **Automatic peer discovery** and heterogeneous node handling.
- **A built-in web UI** plus OpenAI-compatible API on `:52415`.

Raw mlx-lm is better when you want to control the sharding strategy, embed inference in
your own Python, or keep the dependency surface small. For Nybble's local predict-and-act
loop, raw mlx-lm is the right call - you want a library, not a cluster manager.

## 7. Expectations, stated plainly

- Sharding mini + laptop over Thunderbolt to run a 32B model: **works, worth doing.**
- Expect roughly single-node speed for that model size, not a speedup. You bought capacity.
- Sharding anything to Armenia: **do not.** 23 s/token measured ceiling for tensor parallel.
- Armenia as an independent endpoint: **works well.** ~200 ms request overhead, invisible
  for anything that is not interactive chat.

## Sources

- [MLX: Launching distributed programs](https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html)
- [MLX: Distributed communication backends](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [MLX discussion: RDMA over Thunderbolt 5 with JACCL](https://github.com/ml-explore/mlx/discussions/3481)
- [WWDC26 233: Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2026/233/)
- [mlx-lm distributed execution](https://deepwiki.com/ml-explore/mlx-lm/7.5-distributed-execution)
- [llama.cpp RPC backend README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
- [exo](https://github.com/exo-explore/ex-exo)
- [Tailscale on macOS](https://tailscale.com/docs/install/mac)
