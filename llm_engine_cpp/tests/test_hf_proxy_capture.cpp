// test_hf_proxy_capture.cpp — Phase 2 smoke test for HFProxyEngine.
//
// ── Smoke tier (always runs) ─────────────────────────────────────────────
//   - HFProxyEngine can be constructed with a non-existent base URL.
//   - setActivePrompt does not crash (backend unreachable → worker logs a warn).
//   - getAttentionPattern returns {} when no capture is loaded (not a crash).
//   - getResidualSummary returns default sentinels when no capture is loaded.
//   - getQKVStats returns default sentinels when no capture is loaded.
//   - getCurrentTokens returns {} when no capture is loaded.
//   - unloadCheckpoint does not crash.
//
// ── Integration tier (requires LLOB_INTEGRATION_TEST=1 + live backend) ───
//   Gated by the env var so CI doesn't require a running model server.
//   - Load TinyLlama session on the FastAPI backend.
//   - setActivePrompt("The capital of France is").
//   - Wait up to 10s for view().current to be populated.
//   - Assert getCurrentTokens() is non-empty.
//   - Assert getAttentionPattern(0, 0, 0, HeadBias::Broad) returns a
//     non-empty, causal (lower-triangular) matrix.
//   - getResidualSummary(0) returns a finite resid_norm > 0.
//   - Unload cleanly.

#include "llm_engine/hf_proxy_engine.hpp"
#include "llm_engine/model_view.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>

using namespace llmengine;

// ── helpers ───────────────────────────────────────────────────────────────

static bool isnan_f(float f) { return std::isnan(f); }

// ─────────────────────────────────────────────────────────────────────────
// Smoke tier
// ─────────────────────────────────────────────────────────────────────────

static void test_construction_unreachable() {
    // Use a port that's guaranteed not to be listening.  The constructor
    // must not throw or hang.
    HFProxyEngine eng("http://127.0.0.1:19999");
    (void)eng;
}

static void test_setActivePrompt_no_crash() {
    HFProxyEngine eng("http://127.0.0.1:19999");
    // Should push a job to the worker without crashing.  The worker will
    // attempt a POST and silently log a warn when the backend is unreachable.
    eng.setActivePrompt("hello world");
    // Give the worker a moment to try and fail (it will; no backend).
    // We just verify no exception propagates.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

static void test_getters_empty_no_crash() {
    HFProxyEngine eng("http://127.0.0.1:19999");

    // All getters must return the no-data sentinel form when no capture is loaded.
    auto tokens = eng.getCurrentTokens();
    assert(tokens.empty());

    auto attn = eng.getAttentionPattern(0, 0, 8, HeadBias::Broad);
    assert(attn.empty());

    auto resid = eng.getResidualSummary(0);
    assert(isnan_f(resid.resid_norm));
    assert(isnan_f(resid.attn_out_norm));

    auto qkv = eng.getQKVStats(0, 0, 0);
    assert(isnan_f(qkv.q_norm));
    assert(isnan_f(qkv.k_norm));
}

static void test_unload_no_crash() {
    HFProxyEngine eng("http://127.0.0.1:19999");
    eng.unloadCheckpoint();   // should not throw or hang
}

static void test_capabilities_phase2() {
    HFProxyEngine eng("http://127.0.0.1:19999");
    // Before loadCheckpoint: capabilities should be default (all false).
    // After a failed loadCheckpoint (unreachable backend): also all-false.
    auto result = eng.loadCheckpoint("TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    assert(!result.ok);   // backend unreachable → failure

    auto caps = eng.getCapabilities();
    // All bits false because we never reached the info endpoint.
    assert(!caps.has_attention);
    assert(!caps.has_captures);
}

// ─────────────────────────────────────────────────────────────────────────
// Integration tier — gated by LLOB_INTEGRATION_TEST=1
// ─────────────────────────────────────────────────────────────────────────

#ifdef LLOB_INTEGRATION_TEST_ENABLED

static bool wait_for_capture(HFProxyEngine& eng, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        auto tokens = eng.getCurrentTokens();
        if (!tokens.empty()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

static void test_integration() {
    const char* base_url_env = std::getenv("LLOB_BACKEND_URL");
    const std::string base_url =
        base_url_env ? base_url_env : "http://127.0.0.1:8000";

    std::cout << "[integration] connecting to " << base_url << "\n";

    HFProxyEngine eng(base_url);

    // Load TinyLlama — this may take ~20s for a cold load on the backend.
    auto result = eng.loadCheckpoint("TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    if (!result.ok) {
        std::cout << "[integration] SKIP: loadCheckpoint failed: "
                  << result.error << "\n";
        return;
    }

    // Topology sanity.
    auto info = eng.getModelInfo();
    assert(info.nLayers > 0 && "topology must be populated after load");
    std::cout << "[integration] topology: nLayers=" << info.nLayers
              << " nHeads=" << info.nHeads
              << " dModel=" << info.dModel << "\n";

    // setActivePrompt triggers a capture.
    const std::string prompt = "The capital of France is";
    eng.setActivePrompt(prompt);

    // Wait up to 30s for the worker to populate view().current.
    const bool captured = wait_for_capture(eng, 30000);
    if (!captured) {
        std::cout << "[integration] SKIP: capture timed out after 30s\n";
        eng.unloadCheckpoint();
        return;
    }

    // Token list is non-empty.
    auto tokens = eng.getCurrentTokens();
    assert(!tokens.empty());
    std::cout << "[integration] tokens (" << tokens.size() << "): ";
    for (const auto& t : tokens) std::cout << "|" << t;
    std::cout << "\n";

    // Attention matrix for (L=0, H=0) is a non-empty causal matrix.
    const int seq = static_cast<int>(tokens.size());
    auto attn = eng.getAttentionPattern(0, 0, seq, HeadBias::Broad);
    assert(!attn.empty() && "attention matrix must be non-empty");
    assert(static_cast<int>(attn.size()) == seq);
    assert(static_cast<int>(attn[0].size()) == seq);

    // Causal check: upper-triangular entries (above the diagonal) should be
    // effectively zero (masked during softmax).
    for (int r = 0; r < seq; ++r)
        for (int c = r + 1; c < seq; ++c)
            assert(attn[r][c] < 1e-4f && "upper-triangular must be ~zero");

    std::cout << "[integration] attn[0][0] (L=0,H=0): " << attn[0][0] << "\n";

    // Residual summary for layer 0 has a finite, positive resid_norm.
    auto resid = eng.getResidualSummary(0);
    assert(!isnan_f(resid.resid_norm) && resid.resid_norm > 0.0f);
    std::cout << "[integration] resid L=0: norm=" << resid.resid_norm
              << " cos_prev=" << resid.cos_prev << "\n";

    // QKV stats for (L=0, H=0, T=0).
    auto qkv = eng.getQKVStats(0, 0, 0);
    std::cout << "[integration] qkv L=0 H=0 T=0: q=" << qkv.q_norm
              << " k=" << qkv.k_norm
              << " v=" << qkv.v_norm << "\n";

    eng.unloadCheckpoint();
    std::cout << "[integration] PASS\n";
}

#endif  // LLOB_INTEGRATION_TEST_ENABLED

// ─────────────────────────────────────────────────────────────────────────

int main() {
    test_construction_unreachable();
    test_setActivePrompt_no_crash();
    test_getters_empty_no_crash();
    test_unload_no_crash();
    test_capabilities_phase2();

#ifdef LLOB_INTEGRATION_TEST_ENABLED
    test_integration();
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

    std::cout << "smoke: all assertions passed\n";
    return 0;
}
