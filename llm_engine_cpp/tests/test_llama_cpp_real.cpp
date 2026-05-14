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
    REQUIRE(caps.has_state_dict, "has_state_dict after load (parallel GGUF parse)");
    REQUIRE(caps.has_token_stream, "has_token_stream after load (greedy gen loop)");

    // ── State dict (parallel GGUF parse populates view.tensors) ───────────
    // LlamaCppEngine doesn't get tensor enumeration from llama.cpp's public
    // API, so it re-parses the GGUF header in parallel with the llama_model
    // load.  Verify view.tensors is populated and TinyLlama's expected
    // tensors are addressable.
    const auto& tensors = engine.view().tensors;
    REQUIRE(tensors.size() > 0, "view.tensors populated by parallel GGUF parse");
    std::fprintf(stderr, "  state_dict: %zu tensors\n", tensors.size());
    auto state_dict = engine.getStateDict();
    REQUIRE(!state_dict.empty(), "getStateDict returns entries");
    std::fprintf(stderr, "  first tensor: %s (dtype=%s)\n",
                 state_dict.front().name.c_str(),
                 state_dict.front().dtype.c_str());

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

        // ── Token streaming check ──────────────────────────────────────────
        // Prompt was "The capital of France is" = 6 tokens.  After greedy
        // generation, expect at least one more token (the model should
        // produce " Paris" or similar).  Either streaming generated tokens
        // OR EOS hit immediately — verify the former by counting tokens
        // beyond the prompt.
        REQUIRE(toks.size() > 6,
                "streaming generated at least one token past the prompt");
    }

    // ── Head ablation actually mutates the forward pass ──────────────────
    // After setAblation on "blocks.5.attn.head.3", a fresh capture should
    // show that head's attention slice as all zeros — proves cb_eval
    // write-back reached the GPU buffer and the next decode used the
    // masked values.
    REQUIRE(caps.has_intervention,
            "has_intervention after load (head ablation via cb_eval write-back)");
    {
        auto pre_attn = engine.getAttentionPattern(
            5, 3, -1, llmengine::HeadBias::Diag);
        bool pre_has_nonzero = false;
        for (const auto& row : pre_attn)
            for (float v : row)
                if (v != 0.0f) { pre_has_nonzero = true; break; }
        REQUIRE(pre_has_nonzero, "pre-ablation attention[5,3] has non-zero values");

        // Wait for prompt-1 streaming to fully settle so we know any
        // future bundle changes are from prompt 2's ablated decode (not
        // late streaming publishes from prompt 1).  Streaming is done
        // when token_strs stops growing for 1 second.
        std::size_t prev_tokens = 0;
        auto stable_deadline = std::chrono::steady_clock::now()
                             + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < stable_deadline) {
            auto b = engine.view().current.load();
            std::size_t now_tokens = b ? b->token_strs.size() : 0;
            if (now_tokens == prev_tokens && now_tokens > 0) break;
            prev_tokens = now_tokens;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        // Now baseline the bundle pointer — every subsequent change is
        // from the ablated decode.
        auto pre_bundle = engine.view().current.load();

        engine.setAblation({"blocks.5.attn.head.3"}, {});
        engine.setActivePrompt("The capital of France is");

        bool post_capture_ready = false;
        const auto post_deadline = std::chrono::steady_clock::now()
                                 + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < post_deadline) {
            auto b = engine.view().current.load();
            if (b && b != pre_bundle && !b->attention.empty()) {
                post_capture_ready = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        REQUIRE(post_capture_ready, "post-ablation capture appears within 30s");

        if (post_capture_ready) {
            auto post_attn = engine.getAttentionPattern(
                5, 3, -1, llmengine::HeadBias::Diag);
            bool all_zero = !post_attn.empty();
            for (const auto& row : post_attn)
                for (float v : row)
                    if (v != 0.0f) { all_zero = false; break; }
            REQUIRE(all_zero,
                    "post-ablation attention[5,3] is all zeros (cb_eval write-back applied)");

            // Sibling head (5,4) should remain non-zero — verifies ablation
            // is targeted and not a global zero-out.
            auto sibling = engine.getAttentionPattern(
                5, 4, -1, llmengine::HeadBias::Diag);
            bool sibling_has_nonzero = false;
            for (const auto& row : sibling)
                for (float v : row)
                    if (v != 0.0f) { sibling_has_nonzero = true; break; }
            REQUIRE(sibling_has_nonzero,
                    "sibling head (5,4) attention unaffected by ablation of (5,3)");
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
