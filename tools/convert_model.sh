#!/bin/zsh
# convert_model.sh - convert a Hugging Face model to MLX quantized safetensors,
# optionally uploading the result back to the Hub so end users never have to
# convert anything themselves.
#
# This wraps `python3 -m mlx_lm convert`, which does the whole pipeline:
# download the original HF safetensors (bf16/fp16) into ~/.cache/huggingface,
# load, quantize, and write an MLX model directory (config.json + quantized
# *.safetensors + tokenizer files). The output stays plain safetensors - MLX
# "conversion" is quantization plus MLX quantization metadata in config.json,
# not a different container format. Interrupted downloads resume; a re-run with
# the originals already cached skips straight to the quantize step (~1-3 min).
#
# Observed end-to-end times (M-series, ~40 MB/s effective download):
#   Seed-X-PPO-7B  -> 8-bit: ~6 min   (15 GB download dominated)
#   MiLMMT-46-12B  -> 4-bit: ~10 min  (24 GB download dominated)
#
# Usage:
#   tools/convert_model.sh --hf-path <org/name | local-dir> --out <dir>
#                          [--bits 4|8] [--upload-repo <org/name>]
#     --hf-path      source repo id (downloads via huggingface_hub) or a local
#                    directory holding original HF safetensors
#     --out          destination directory for the MLX model (must not exist;
#                    mlx_lm refuses to overwrite)
#     --bits         quantization bits (default 4). For a fixed RAM budget,
#                    more params @ 4-bit beats fewer @ 8-bit
#     --upload-repo  also create/push the converted model to this Hub repo
#                    (needs `hf auth login` with a WRITE token first; mlx_lm
#                    creates the repo and a model card crediting the original).
#                    Community naming convention: <ModelName>-4bit / -8bit
#
# Gated/licensed sources: a gated repo needs `hf auth login` even for download.
# When REDISTRIBUTING a conversion (--upload-repo), the original model's license
# obligations bind YOU as the distributor - e.g. a Gemma-family model (gemma
# license tag: TranslateGemma, MiLMMT-46) requires shipping the "Gemma is
# provided under and subject to the Gemma Terms of Use ..." notice and carrying
# the use restrictions; set the card's `license:` metadata to match the
# original. Check the source repo's license before uploading.

set -e
set -o pipefail

BITS=4
HF_PATH=""
OUT=""
UPLOAD=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-path)     HF_PATH="$2"; shift 2 ;;
        --out)         OUT="$2"; shift 2 ;;
        --bits)        BITS="$2"; shift 2 ;;
        --upload-repo) UPLOAD="$2"; shift 2 ;;
        *) echo "unknown argument: $1 (see header for usage)"; exit 2 ;;
    esac
done
[[ -n "$HF_PATH" && -n "$OUT" ]] || { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[[ "$BITS" == "4" || "$BITS" == "8" ]] || { echo "--bits must be 4 or 8"; exit 2; }
[[ ! -e "$OUT" ]] || { echo "$OUT already exists; mlx_lm convert refuses to overwrite - move it aside first"; exit 1; }

python3 -c "import mlx_lm" 2>/dev/null || {
    echo "mlx_lm not importable; install with: python3 -m pip install -U mlx-lm"; exit 1
}
echo "[convert] mlx-lm $(python3 -c "import mlx_lm; print(mlx_lm.__version__)")"
echo "[convert] $HF_PATH -> $OUT (${BITS}-bit)${UPLOAD:+, then upload to $UPLOAD}"

ARGS=(--hf-path "$HF_PATH" --mlx-path "$OUT" -q --q-bits "$BITS")
[[ -n "$UPLOAD" ]] && ARGS+=(--upload-repo "$UPLOAD")
python3 -m mlx_lm convert "${ARGS[@]}"

echo "[convert] done: $(du -sh "$OUT" | cut -f1) in $OUT"
echo "[convert] originals cached under ~/.cache/huggingface/hub (delete when finished)"
