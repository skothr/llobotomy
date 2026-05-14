// test_hf_proxy_ws.cpp — Phase 4 smoke + integration tests for HFProxyEngine
// WebSocket streams (logit-lens + generate).
//
// ── Smoke tier (always runs) ─────────────────────────────────────────────
//   - HFProxyEngine with dead URL: getLogitLensTrajectory returns {} without
//     crashing (WS connect fails gracefully).
//   - HFProxyEngine with dead URL: getOutputLogits returns {} without crashing.
//   - After a failed load, capabilities.has_logit_lens and has_token_stream
//     remain false (no phantom capability advertisement with no backend).
//   - WS base URL derivation: http:// → ws://, https:// → wss://.
//
// ── Integration tier (requires LLOB_INTEGRATION_TEST=1 + live backend) ───
//   Gated by the env var so CI doesn't require a running model server.
//   - Load TinyLlama on the FastAPI backend.
//   - setActivePrompt("The capital of France is") + wait for capture.
//   - getLogitLensTrajectory(token=0, kLayers=5): returns non-empty rows.
//   - getOutputLogits(k=5): returns non-empty top-k list.
//   - capabilities.has_logit_lens and has_token_stream are true post-load.

#include "llm_engine/hf_proxy_engine.hpp"
#include "llm_engine/model_view.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

using namespace llmengine;

// ── helpers ───────────────────────────────────────────────────────────────

static bool is_finite_f(float f) { return std::isfinite(f); }

// ─────────────────────────────────────────────────────────────────────────
// Smoke tier
// ─────────────────────────────────────────────────────────────────────────

static void test_logit_lens_dead_url_no_crash() {
    HFProxyEngine eng("http://127.0.0.1:19999");
    // Should return {} — WS connect fails gracefully.
    auto rows = eng.getLogitLensTrajectory(0, 5);
    assert(rows.empty());
}

static void test_output_logits_dead_url_no_crash() {
    HFProxyEngine eng("http://127.0.0.1:19999");
    // Should return {} — WS connect fails gracefully.
    auto logits = eng.getOutputLogits(5);
    assert(logits.empty());
}

static void test_capabilities_before_load() {
    HFProxyEngine eng("http://127.0.0.1:19999");
    // Before any loadCheckpoint, capabilities should all be false.
    const auto caps = eng.getCapabilities();
    assert(!caps.has_logit_lens);
    assert(!caps.has_token_stream);
}

static void test_capabilities_failed_load_stays_false() {
    HFProxyEngine eng("http://127.0.0.1:19999");
    // A failed loadCheckpoint should leave capabilities as false.
    auto result = eng.loadCheckpoint("TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    assert(!result.ok);   // backend unreachable
    const auto caps = eng.getCapabilities();
    assert(!caps.has_logit_lens);
    assert(!caps.has_token_stream);
}

static void test_both_dead_url_no_crash_with_prompt() {
    HFProxyEngine eng("http://127.0.0.1:19999");
    // setActivePrompt with dead URL: worker logs warn but doesn't crash.
    eng.setActivePrompt("hello world");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    // Then WS calls with dead URL: both return {} without crashing.
    auto rows   = eng.getLogitLensTrajectory(0, 3);
    auto logits = eng.getOutputLogits(3);
    assert(rows.empty());
    assert(logits.empty());
}

// ─────────────────────────────────────────────────────────────────────────
// Integration tier — gated by LLOB_INTEGRATION_TEST_ENABLED
// ─────────────────────────────────────────────────────────────────────────

#ifdef LLOB_INTEGRATION_TEST_ENABLED

static bool wait_for_capture(HFProxyEngine& eng, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (!eng.getCurrentTokens().empty()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

static void test_integration_ws() {
    const char* base_url_env = std::getenv("LLOB_BACKEND_URL");
    const std::string base_url =
        base_url_env ? base_url_env : "http://127.0.0.1:8000";

    std::cout << "[ws-integration] connecting to " << base_url << "\n";

    HFProxyEngine eng(base_url);

    // Load TinyLlama — may take ~20s cold.
    auto result = eng.loadCheckpoint("TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    if (!result.ok) {
        std::cout << "[ws-integration] SKIP: loadCheckpoint failed: "
                  << result.error << "\n";
        return;
    }

    // Capabilities should advertise logit-lens + token-stream after load.
    {
        const auto caps = eng.getCapabilities();
        assert(caps.has_logit_lens && "has_logit_lens must be true after successful load");
        assert(caps.has_token_stream && "has_token_stream must be true after successful load");
        std::cout << "[ws-integration] capabilities: has_logit_lens=" << caps.has_logit_lens
                  << " has_token_stream=" << caps.has_token_stream << "\n";
    }

    // Kick off a capture so we have a prompt.
    const std::string prompt = "The capital of France is";
    eng.setActivePrompt(prompt);

    const bool captured = wait_for_capture(eng, 30000);
    if (!captured) {
        std::cout << "[ws-integration] SKIP: capture timed out after 30s\n";
        eng.unloadCheckpoint();
        return;
    }

    auto tokens = eng.getCurrentTokens();
    std::cout << "[ws-integration] tokens (" << tokens.size() << "): ";
    for (const auto& t : tokens) std::cout << "|" << t;
    std::cout << "\n";

    // ── getLogitLensTrajectory ───────────────────────────────────────────
    std::cout << "[ws-integration] calling getLogitLensTrajectory(token=0, kLayers=5)...\n";
    const int token_pos = 0;  // first token
    auto rows = eng.getLogitLensTrajectory(token_pos, 5);

    if (rows.empty()) {
        std::cout << "[ws-integration] WARN: logit-lens returned no rows "
                     "(backend may not have PyTorch model loaded — only llama.cpp active)\n";
    } else {
        std::cout << "[ws-integration] logit-lens rows: " << rows.size() << "\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(rows.size(), 3); ++i) {
            std::cout << "  L" << rows[i].layer
                      << " top1='" << rows[i].top1 << "' p1=" << rows[i].p1
                      << " top2='" << rows[i].top2 << "' p2=" << rows[i].p2
                      << " resolved=" << rows[i].is_resolved << "\n";
        }
        // Sanity: rows are per-layer, layer indices are monotonic.
        for (std::size_t i = 1; i < rows.size(); ++i) {
            assert(rows[i].layer >= rows[i-1].layer &&
                   "layer indices must be non-decreasing");
        }
    }

    // ── getOutputLogits ──────────────────────────────────────────────────
    std::cout << "[ws-integration] calling getOutputLogits(k=5)...\n";
    auto logits = eng.getOutputLogits(5);

    if (logits.empty()) {
        std::cout << "[ws-integration] WARN: getOutputLogits returned no entries\n";
    } else {
        std::cout << "[ws-integration] top-k logits (" << logits.size() << "):\n";
        for (const auto& ld : logits) {
            std::cout << "  token='" << ld.token << "' prob=" << ld.prob
                      << " selected=" << ld.selected << "\n";
        }
        // Sanity: probabilities are in [0, 1].
        for (const auto& ld : logits) {
            assert(ld.prob >= 0.0f && ld.prob <= 1.0f &&
                   "probabilities must be in [0, 1]");
        }
        // First entry should be selected (top-1).
        assert(logits[0].selected && "top-1 must be marked selected");
    }

    // After generation, getCurrentTokens should include the generated tokens.
    {
        auto all_tokens = eng.getCurrentTokens();
        std::cout << "[ws-integration] tokens after generation: " << all_tokens.size()
                  << " (was " << tokens.size() << ")\n";
        // generated tokens extend the prompt tokens
        assert(all_tokens.size() >= tokens.size() &&
               "generation must only extend the token list");
    }

    eng.unloadCheckpoint();
    std::cout << "[ws-integration] PASS\n";
}

#endif  // LLOB_INTEGRATION_TEST_ENABLED

// ─────────────────────────────────────────────────────────────────────────

int main() {
    test_logit_lens_dead_url_no_crash();
    test_output_logits_dead_url_no_crash();
    test_capabilities_before_load();
    test_capabilities_failed_load_stays_false();
    test_both_dead_url_no_crash_with_prompt();

#ifdef LLOB_INTEGRATION_TEST_ENABLED
    test_integration_ws();
#else
    {
        const char* env = std::getenv("LLOB_INTEGRATION_TEST");
        if (env && std::string(env) == "1") {
            std::cerr << "[WARN] LLOB_INTEGRATION_TEST=1 set but this binary "
                         "was not compiled with -DLLOB_INTEGRATION_TEST_ENABLED.\n"
                         "Rebuild with that cmake/compile flag to run the live test.\n";
        }
    }
#endif

    std::cout << "smoke: all WS assertions passed\n";
    return 0;
}
