// test_llama_cpp_real.cpp — LlamaCppEngine deep tier.
//
// Requires env var LLOB_DEEP_LLAMA_GGUF_PATH to point at a real GGUF file.
// When the var is absent, exits 0 silently (self-skip).
//
// When present:
//   1. Loads the model and verifies topology fields are populated.
//   2. Submits "The capital of France is" and waits up to 10s for the
//      engine worker to complete the forward pass.
//   3. Verifies view().topology.nLayers > 0 (real model reported its depth).
//   4. Verifies getAttentionPattern(0, 0, ...) returns a non-empty causal
//      matrix (at least 1 row).
//
// This test does NOT require llama.cpp support to compile — it skips
// gracefully at runtime when the binary was built without it.

#include "llm_engine/llama_cpp_engine.hpp"
#include "llm_engine/model_view.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

static int g_fail = 0;
#define REQUIRE(cond, msg) do {                                         \
    if (!(cond)) {                                                      \
        std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__,  \
                     (msg));                                            \
        ++g_fail;                                                       \
    } else {                                                            \
        std::fprintf(stdout, "PASS  %s\n", (msg));                      \
    }                                                                   \
} while(0)

int main()
{
    const char* gguf_path = std::getenv("LLOB_DEEP_LLAMA_GGUF_PATH");
    if (!gguf_path || gguf_path[0] == '\0') {
        std::fprintf(stdout,
            "[llama_cpp_real] LLOB_DEEP_LLAMA_GGUF_PATH not set — skip.\n");
        return 0;   // self-skip
    }

#ifndef LLM_ENGINE_HAVE_LLAMA_CPP
    std::fprintf(stdout,
        "[llama_cpp_real] Built without llama.cpp support — skip.\n");
    return 0;
#else
    std::fprintf(stdout, "=== LlamaCppEngine deep test ===\n");
    std::fprintf(stdout, "GGUF: %s\n\n", gguf_path);

    llmengine::LlamaCppEngine engine;

    // ── Load ─────────────────────────────────────────────────────────────
    auto r = engine.loadCheckpoint(gguf_path);
    if (!r.ok) {
        std::fprintf(stderr, "loadCheckpoint FAILED: %s\n", r.error.c_str());
        return 1;
    }
    REQUIRE(r.ok, "loadCheckpoint returns ok=true");

    // ── Topology ──────────────────────────────────────────────────────────
    const auto& topo = engine.view().topology;
    std::fprintf(stdout, "  nLayers=%d  nHeads=%d  dModel=%d  vocab=%d\n",
                 topo.nLayers, topo.nHeads, topo.dModel, topo.vocab);
    REQUIRE(topo.nLayers > 0, "topology.nLayers > 0");
    REQUIRE(topo.nHeads  > 0, "topology.nHeads > 0");
    REQUIRE(topo.dModel  > 0, "topology.dModel > 0");
    REQUIRE(topo.vocab   > 0, "topology.vocab  > 0");

    // ── Capabilities ──────────────────────────────────────────────────────
    auto caps = engine.getCapabilities();
    REQUIRE(caps.has_topology, "has_topology after load");
    REQUIRE(caps.has_attention, "has_attention after load");

    // ── Forward pass ──────────────────────────────────────────────────────
    engine.setActivePrompt("The capital of France is");

    // Wait up to 15s for the capture to populate.
    bool capture_ready = false;
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::seconds(15);
    while (std::chrono::steady_clock::now() < deadline) {
        auto bundle = engine.view().current.load();
        if (bundle && !bundle->attention.empty()) {
            capture_ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    REQUIRE(capture_ready, "capture populates within 15s");

    if (capture_ready) {
        // ── Attention check ────────────────────────────────────────────────
        auto attn = engine.getAttentionPattern(0, 0, -1,
                                               llmengine::HeadBias::Diag);
        REQUIRE(!attn.empty(), "getAttentionPattern(0,0) returns non-empty matrix");
        if (!attn.empty()) {
            std::fprintf(stdout,
                "  attention[0,0] shape: %zu x %zu\n",
                attn.size(), attn.empty() ? 0UZ : attn[0].size());
        }

        // ── Token strings check ────────────────────────────────────────────
        auto toks = engine.getCurrentTokens();
        REQUIRE(!toks.empty(), "getCurrentTokens returns non-empty after capture");
        if (!toks.empty()) {
            std::fprintf(stdout, "  tokens (%zu): ", toks.size());
            for (const auto& t : toks) std::fprintf(stdout, "[%s]", t.c_str());
            std::fprintf(stdout, "\n");
        }
    }

    // ── Drain logs ────────────────────────────────────────────────────────
    auto logs = engine.drainEngineLogs();
    if (!logs.empty()) {
        std::fprintf(stdout, "  engine logs (%zu):\n", logs.size());
        for (const auto& l : logs)
            std::fprintf(stdout, "    [%s] %s\n",
                l.sev == llmengine::Severity::Error ? "ERR"
              : l.sev == llmengine::Severity::Warn  ? "WRN" : "INF",
                l.msg.c_str());
    }

    // ── Unload ────────────────────────────────────────────────────────────
    engine.unloadCheckpoint();
    REQUIRE(engine.view().topology.nLayers == llmengine::kNoInt,
            "post-unload nLayers is sentinel");

    if (g_fail == 0) {
        std::fprintf(stdout, "\nAll deep tests PASSED.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d deep test(s) FAILED.\n", g_fail);
    return 1;
#endif  // LLM_ENGINE_HAVE_LLAMA_CPP
}
