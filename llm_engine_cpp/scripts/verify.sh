#!/usr/bin/env bash
#
# verify.sh — full regression verification for llm_engine.
#
# Runs the whole gate: both build modes (mock-data ON/OFF) build clean,
# smoke tests pass on both, deep tests run-skip-or-pass on both (deep
# tests are env-gated and skip-friendly when fixtures are absent).
#
# Exits 0 on full pass. Non-zero on any build failure or smoke
# failure. Deep-tier *skips* are not failures.
#
# Run from anywhere — script resolves its own dir.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'

ok()   { printf "${GREEN}✓${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}⚠${RESET} %s\n" "$*"; }
err()  { printf "${RED}✗${RESET} %s\n" "$*" >&2; }

run_mode() {
    local mode="$1"           # "ON" | "OFF" — controls LLM_ENGINE_USE_MOCK_DATA
    local llama="${2:-OFF}"   # "ON" | "OFF" — controls LLM_ENGINE_BUILD_LLAMA_CPP
    local tag="${mode,,}"
    if [[ "$llama" == "ON" ]]; then tag="${tag}-llama"; fi
    local build_dir="${ENGINE_DIR}/build-verify-${tag}"
    echo
    echo "── Build mode: LLM_ENGINE_USE_MOCK_DATA=${mode}  LLAMA_CPP=${llama} ──"

    cmake -S "$ENGINE_DIR" -B "$build_dir" \
          -DLLM_ENGINE_USE_MOCK_DATA="$mode" \
          -DLLM_ENGINE_BUILD_LLAMA_CPP="$llama" \
          -DLLM_ENGINE_BUILD_TESTS=ON \
          > "${build_dir}-configure.log" 2>&1 \
        || { err "cmake configure failed (${mode}/${llama})"; cat "${build_dir}-configure.log"; return 1; }
    ok "configured (${mode}/${llama})"

    cmake --build "$build_dir" -j > "${build_dir}-build.log" 2>&1 \
        || { err "build failed (${mode}/${llama})"; tail -50 "${build_dir}-build.log"; return 1; }
    ok "built (${mode}/${llama})"

    echo "  Smoke tests:"
    ctest --test-dir "$build_dir" -L smoke --output-on-failure 2>&1 \
        | sed 's/^/    /' \
        || { err "smoke tests failed (${mode}/${llama})"; return 1; }
    ok "smoke passed (${mode}/${llama})"

    echo "  Deep tests (env-gated; skips are not failures):"
    # Note: deep tests should self-skip with exit 0 when their required
    # fixtures / env vars are absent. We still fail this script on a
    # real deep-test failure.
    #
    # Detection: `ctest -N -L deep` lists matching tests; the trailing
    # summary line "Total Tests: N" tells us whether any are registered.
    local deep_count
    deep_count=$(ctest --test-dir "$build_dir" -N -L deep 2>/dev/null \
                 | sed -n 's/^Total Tests: \([0-9]\+\)/\1/p')
    if [[ "${deep_count:-0}" -gt 0 ]]; then
        ctest --test-dir "$build_dir" -L deep --output-on-failure 2>&1 \
            | sed 's/^/    /' \
            || { err "deep tests failed (${mode}/${llama})"; return 1; }
        ok "deep ran (${mode}/${llama}) — ${deep_count} test(s)"
    else
        warn "deep tier has no registered tests (yet)"
    fi
}

echo "llm_engine verify.sh"
echo "Engine dir: $ENGINE_DIR"

# Default sweep: 2 modes (MOCK_DATA ON/OFF) × LLAMA_CPP=OFF.
# Pass `--llama` as a flag to also run the LLAMA_CPP=ON pair (needs
# llama.cpp pre-built at /home/ai/ai-projects/llm/lib/llama.cpp).
INCLUDE_LLAMA=0
for arg in "$@"; do
    case "$arg" in
        --llama|--with-llama) INCLUDE_LLAMA=1 ;;
    esac
done

run_mode ON  OFF
run_mode OFF OFF
if [[ "$INCLUDE_LLAMA" == "1" ]]; then
    run_mode ON  ON
    run_mode OFF ON
fi

echo
ok "verify.sh complete — all configurations pass smoke; deep tier ran where applicable"
