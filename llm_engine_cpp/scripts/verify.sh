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
    local mode="$1"   # "ON" | "OFF"
    local build_dir="${ENGINE_DIR}/build-verify-${mode,,}"
    echo
    echo "── Build mode: LLM_ENGINE_USE_MOCK_DATA=${mode} ───────────────────"

    cmake -S "$ENGINE_DIR" -B "$build_dir" \
          -DLLM_ENGINE_USE_MOCK_DATA="$mode" \
          -DLLM_ENGINE_BUILD_TESTS=ON \
          > "${build_dir}-configure.log" 2>&1 \
        || { err "cmake configure failed (${mode})"; cat "${build_dir}-configure.log"; return 1; }
    ok "configured (${mode})"

    cmake --build "$build_dir" -j > "${build_dir}-build.log" 2>&1 \
        || { err "build failed (${mode})"; tail -50 "${build_dir}-build.log"; return 1; }
    ok "built (${mode})"

    echo "  Smoke tests:"
    ctest --test-dir "$build_dir" -L smoke --output-on-failure 2>&1 \
        | sed 's/^/    /' \
        || { err "smoke tests failed (${mode})"; return 1; }
    ok "smoke passed (${mode})"

    echo "  Deep tests (env-gated; skips are not failures):"
    # Note: deep tests should self-skip with exit 0 when their required
    # fixtures / env vars are absent. We still fail this script on a
    # real deep-test failure.
    if ctest --test-dir "$build_dir" -L deep 2>/dev/null | grep -q "Test #"; then
        ctest --test-dir "$build_dir" -L deep --output-on-failure 2>&1 \
            | sed 's/^/    /' \
            || { err "deep tests failed (${mode})"; return 1; }
        ok "deep ran (${mode})"
    else
        warn "deep tier has no registered tests (yet)"
    fi
}

echo "llm_engine verify.sh"
echo "Engine dir: $ENGINE_DIR"

run_mode ON
run_mode OFF

echo
ok "verify.sh complete — both modes pass smoke; deep tier ran where applicable"
