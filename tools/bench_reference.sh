#!/bin/zsh
# bench_reference.sh - run mlx-agent's bench mode with the reference model from
# Apple's M5 announcement, on whatever Mac this is, and append one JSON line to
# a shared results file. Run it on each machine (M1 mini, M1 Pro, M2 Air, M5 Air)
# and diff the lines.
#
# Reference model: mlx-community/Qwen3-8B-4bit - one of the models in Apple's
# "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU" post
# (machinelearning.apple.com/research/exploring-llms-mlx-m5). Methodology matched:
# 4096-token prompt, 128 generated tokens, TTFT + generation tok/s.
#
# Usage:
#   tools/bench_reference.sh [model-dir] [results-file]
#     model-dir     default: the HF cache copy of mlx-community/Qwen3-8B-4bit,
#                   downloaded on first run (~4.3 GB; needs python3 + huggingface_hub
#                   OR the hf CLI)
#     results-file  default: ~/mlx-bench-results.jsonl
#
# IMPORTANT for comparable numbers:
#   - plug the machine in, close GPU-hungry apps
#   - let it cool down after any big build/compile: a fanless Air throttles the
#     GPU for minutes after sustained CPU load (observed: ~40% prefill loss)

set -e
set -o pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="${1:-}"
RESULTS="${2:-$HOME/mlx-bench-results.jsonl}"
REPO="mlx-community/Qwen3-8B-4bit"

if [[ -z "$MODEL_DIR" ]]; then
    echo "[bench] resolving $REPO (downloads ~4.3 GB on first run)..."
    MODEL_DIR=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$REPO'))
" 2>/dev/null) || {
        echo "[bench] python3/huggingface_hub not available; trying hf CLI..."
        MODEL_DIR=$(hf download "$REPO" | tail -1)
    }
fi
[[ -f "$MODEL_DIR/config.json" ]] || { echo "no config.json in $MODEL_DIR"; exit 1; }
echo "[bench] model: $MODEL_DIR"

BIN=build/Build/Products/Release/mlx-agent
if [[ ! -x "$BIN" ]]; then
    echo "[bench] building mlx-agent (Release, xcodebuild - see README)..."
    xcodebuild -scheme mlx-agent -destination 'platform=macOS,arch=arm64' \
        -derivedDataPath build -configuration Release \
        -skipPackagePluginValidation -skipMacroValidation build | tail -1
fi

OUT=$("$BIN" bench --model "$MODEL_DIR" "${@[3,-1]}")
echo "$OUT"

LINE=$(echo "$OUT" | sed -n 's/^RESULT_JSON: //p')
if [[ -n "$LINE" ]]; then
    HOST=$(scutil --get ComputerName 2>/dev/null || hostname)
    # Host goes in via the environment, not interpolated into Python source: a name
    # containing quotes or backslashes must not be able to break the JSON line.
    echo "$LINE" | BENCH_HOST="$HOST" python3 -c "
import json, os, sys
r = json.load(sys.stdin)
r['host'] = os.environ['BENCH_HOST']
print(json.dumps(r, sort_keys=True))
" >> "$RESULTS"
    echo "[bench] appended to $RESULTS"
fi
