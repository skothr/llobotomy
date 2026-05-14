#!/usr/bin/env bash
#
# demo.sh — one-command end-to-end demo of the llm_engine + gui_cpp stack.
#
# Defaults to TinyLlama-1.1B-Chat-v1.0 (cached at testing/.cache/models/).
# Override the HF model via TINYLLAMA_HF_DIR, the GGUF output path via
# DEMO_GGUF, or the binary backend via LLOB_BACKEND (default: llama_cpp).
#
# Exits 0 on launch.  The GUI runs in the foreground; close it normally.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GUI_DIR="$(cd "$ENGINE_DIR/../gui_cpp" && pwd)"
REPO_ROOT="$(cd "$ENGINE_DIR/../.." && pwd)"

# Defaults.
: "${TINYLLAMA_HF_DIR:=$REPO_ROOT/testing/.cache/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0}"
: "${DEMO_GGUF:=/tmp/tinyllama-1.1b-chat-f16.gguf}"
: "${LLOB_BACKEND:=llama_cpp}"
: "${ENGINE_BUILD_DIR:=$ENGINE_DIR/build-demo}"
: "${GUI_BUILD_DIR:=$GUI_DIR/build}"

GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
say() { printf "${GREEN}=>${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}!!${RESET} %s\n" "$*"; }

# 1. Make sure the GGUF exists (convert from HF on demand).
if [[ ! -f "$DEMO_GGUF" ]]; then
    say "Converting HF model -> GGUF (one-time, ~30s on a recent box)"
    if [[ ! -d "$TINYLLAMA_HF_DIR" ]]; then
        warn "HF model dir not found: $TINYLLAMA_HF_DIR"
        warn "Set TINYLLAMA_HF_DIR or download the model first."
        exit 1
    fi
    SNAPSHOT="$(find "$TINYLLAMA_HF_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)"
    if [[ -z "$SNAPSHOT" ]]; then
        warn "No snapshot found under $TINYLLAMA_HF_DIR/snapshots"
        exit 1
    fi
    python "$REPO_ROOT/lib/llama.cpp/convert_hf_to_gguf.py" \
        "$SNAPSHOT" \
        --outtype f16 \
        --outfile "$DEMO_GGUF"
fi
say "GGUF ready: $DEMO_GGUF"

# 2. Build the engine with LlamaCpp enabled.
say "Configuring engine build (LLAMA_CPP=ON, MOCK_DATA=OFF)"
cmake -S "$ENGINE_DIR" -B "$ENGINE_BUILD_DIR" \
    -DLLM_ENGINE_USE_MOCK_DATA=OFF \
    -DLLM_ENGINE_BUILD_LLAMA_CPP=ON \
    > /dev/null
say "Building engine"
cmake --build "$ENGINE_BUILD_DIR" -j > /dev/null

# 3. Build the GUI (which links the engine).
say "Building gui_cpp"
if [[ ! -L "$GUI_DIR/libs/imgui" && ! -d "$GUI_DIR/libs/imgui" ]]; then
    warn "gui_cpp/libs/imgui missing - see gui_cpp setup notes."
    exit 1
fi
cmake -S "$GUI_DIR" -B "$GUI_BUILD_DIR" -DLLOB_USE_MOCK_DATA=OFF > /dev/null
cmake --build "$GUI_BUILD_DIR" -j > /dev/null

# 4. Launch.
say "Launching llobotomy (backend=$LLOB_BACKEND, gguf=$DEMO_GGUF)"
say "  Inference workspace: type a prompt and hit Submit"
say "  Attention workspace: click head grid cells to ablate"
say "  Sampler controls live under the prompt input"
exec env \
    LLOB_BACKEND="$LLOB_BACKEND" \
    LLOB_GGUF_PATH="$DEMO_GGUF" \
    "$GUI_BUILD_DIR/llobotomy" "$@"
