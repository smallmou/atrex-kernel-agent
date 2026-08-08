#!/usr/bin/env bash
#
# Launch one optimization campaign per FlashInfer-Bench operator.
#
# --framework takes a single value, so each operator gets one campaign per framework.
# Correctness/performance for these SOL-ExecBench ops is scored by the sol-execbench
# evaluator through the workspace's test_kernel.py (see reference/sol_seed.py).
#
# Usage:
#   scripts/run_flashinfer_bench.sh [--dry-run] [operator-name ...]
#
# With no operator names, every operator under BENCH_DIR is launched.

set -euo pipefail

BENCH_DIR="${BENCH_DIR:-/home/liangyan/SOL-ExecBench/data/benchmark/FlashInfer-Bench}"
WORKSPACE="${WORKSPACE:-/home/liangyan/aka-opt-sol-flashinfer}"
PLATFORM="${PLATFORM:-B200}"
SANDBOX_HW="${SANDBOX_HW:-L20C}"
MAX_ITERS="${MAX_ITERS:-100}"
AGENT_CLI="${AGENT_CLI:-claude}"
FRAMEWORKS="${FRAMEWORKS:-Cuda}"
# Per-iteration hang backstop. Under many concurrent campaigns every sandbox round-trip queues
# behind other gateway jobs, so a session needing many round-trips can outlive the orchestrator's
# 90-minute default and be killed mid-iteration. Empty = keep the orchestrator's default.
ITER_TIMEOUT="${ITER_TIMEOUT:-}"
# Peak-utilization short-circuit. The orchestrator stops a campaign once its estimated peak
# utilization crosses this, but that estimate is a roofline proxy rather than the objective
# (geomean latency), so it can end a campaign with most of --max-iters unused. Empty = keep the
# orchestrator's default of 90.
TARGET_UTIL="${TARGET_UTIL:-}"
# The orchestrator does not throttle gateway submissions, so space the V0 benches out
# rather than firing every campaign's first job in the same instant.
STAGGER_S="${STAGGER_S:-20}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

# claude lives under nvm, which is not on a non-interactive PATH.
if ! command -v claude >/dev/null 2>&1; then
    for candidate in /root/.nvm/versions/node/*/bin; do
        [[ -x "$candidate/claude" ]] && export PATH="$candidate:$PATH" && break
    done
fi
if [[ "$AGENT_CLI" == "claude" ]] && ! command -v claude >/dev/null 2>&1; then
    echo "error: --agent-cli claude requested but claude is not on PATH" >&2
    exit 1
fi

mkdir -p "$WORKSPACE/logs"

if [[ $# -gt 0 ]]; then
    OP_DIRS=()
    for op in "$@"; do
        [[ -d "$BENCH_DIR/$op" ]] || { echo "error: no such operator: $op" >&2; exit 1; }
        OP_DIRS+=("$BENCH_DIR/$op")
    done
else
    mapfile -t OP_DIRS < <(find "$BENCH_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
fi
if [[ ${#OP_DIRS[@]} -eq 0 ]]; then
    echo "error: no operator dirs under $BENCH_DIR" >&2
    exit 1
fi

echo "operators : ${#OP_DIRS[@]}"
echo "frameworks: $FRAMEWORKS"
echo "campaigns : $(( ${#OP_DIRS[@]} * $(wc -w <<<"$FRAMEWORKS") ))"
echo "workspace : $WORKSPACE"
echo "target    : $PLATFORM via $SANDBOX_HW, max-iters=$MAX_ITERS, agent-cli=$AGENT_CLI"
echo

launched=0
for op_dir in "${OP_DIRS[@]}"; do
    op="$(basename "$op_dir")"
    for fw in $FRAMEWORKS; do
        log="$WORKSPACE/logs/${op}_${fw,,}.log"
        cmd=(python -u "$REPO_ROOT/orchestrator/optimize.py"
             --op-dir "$op_dir"
             --platform "$PLATFORM"
             --sandbox-hardware "$SANDBOX_HW"
             --max-iters "$MAX_ITERS"
             --agent-cli "$AGENT_CLI"
             --framework "$fw"
             --workspace "$WORKSPACE")
        [[ -n "$ITER_TIMEOUT" ]] && cmd+=(--iter-timeout "$ITER_TIMEOUT")
        [[ -n "$TARGET_UTIL" ]] && cmd+=(--target-util "$TARGET_UTIL")
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "${cmd[*]} > $log"
            continue
        fi
        ( cd "$REPO_ROOT" && printf '\n===== launch %s %s =====\n' "$op/$fw" "$(date '+%F %T')" >> "$log" \
          && setsid nohup "${cmd[@]}" >> "$log" 2>&1 < /dev/null & echo "$op/$fw pid=$!" )
        launched=$((launched + 1))
        sleep "$STAGGER_S"
    done
done

[[ $DRY_RUN -eq 1 ]] && exit 0
echo
echo "launched $launched campaigns; logs under $WORKSPACE/logs/"
