// test_llama_cpp_smoke.cpp — LlamaCppEngine smoke tier.
//
// Exercises the engine without a real model file.  When compiled without
// LLM_ENGINE_HAVE_LLAMA_CPP (the off build), every test verifies the
// graceful-failure path.  When compiled with the flag, tests verify
// construction, bad-path failure, and post-unload sentinels.
//
// All tests return 0 (pass) regardless of the llama.cpp build flag so
// the smoke suite never blocks the default (OFF) build.

#include "llm_engine/llama_cpp_engine.hpp"
#include "llm_engine/model_view.hpp"

#include <cassert>
#include <cstdio>
#include <string>

// Minimal assertion helper.
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

namespace {

// ── Test 1: construction doesn't crash ───────────────────────────────────────
void test_construct()
{
    llmengine::LlamaCppEngine engine;
    // Verify sentinels before any load.
    REQUIRE(engine.view().topology.nLayers == llmengine::kNoInt,
            "pre-load nLayers is sentinel");
    REQUIRE(engine.view().topology.nHeads  == llmengine::kNoInt,
            "pre-load nHeads is sentinel");
    REQUIRE(engine.view().current.load() == nullptr,
            "pre-load current is null");
}

// ── Test 2: bad path returns {ok=false} ──────────────────────────────────────
void test_bad_path()
{
    llmengine::LlamaCppEngine engine;
    auto r = engine.loadCheckpoint("/nonexistent/path/model.gguf");
    REQUIRE(!r.ok,    "bad path returns ok=false");
    REQUIRE(!r.error.empty(), "bad path has non-empty error string");
    std::fprintf(stdout, "  error message: %s\n", r.error.c_str());
}

// ── Test 3: getAttentionPattern before load returns {} ───────────────────────
void test_attn_before_load()
{
    llmengine::LlamaCppEngine engine;
    auto attn = engine.getAttentionPattern(0, 0, 8,
                                           llmengine::HeadBias::Diag);
    REQUIRE(attn.empty(), "getAttentionPattern before load returns empty");
}

// ── Test 4: getCurrentTokens before load returns {} ──────────────────────────
void test_tokens_before_load()
{
    llmengine::LlamaCppEngine engine;
    auto toks = engine.getCurrentTokens();
    REQUIRE(toks.empty(), "getCurrentTokens before load returns empty");
}

// ── Test 5: capabilities bitmap ──────────────────────────────────────────────
void test_capabilities()
{
    llmengine::LlamaCppEngine engine;
    auto caps = engine.getCapabilities();
    REQUIRE(caps.has_attention,  "has_attention advertised");
    REQUIRE(caps.has_residual,   "has_residual advertised (Wave D extensions)");
    REQUIRE(caps.has_logit_lens, "has_logit_lens advertised (Wave D extensions — logits captured)");
    REQUIRE(caps.has_topology,   "has_topology advertised");
    REQUIRE(!caps.has_intervention, "has_intervention not advertised");
    REQUIRE(!caps.has_token_stream, "has_token_stream not advertised");
}

// ── Test 6: view() topology clears after unload ───────────────────────────────
void test_unload_clears_view()
{
    llmengine::LlamaCppEngine engine;
    // Even without a successful load, calling unload should not crash.
    engine.unloadCheckpoint();
    REQUIRE(engine.view().topology.nLayers == llmengine::kNoInt,
            "post-unload nLayers is sentinel");
    REQUIRE(engine.view().current.load() == nullptr,
            "post-unload current is null");
}

// ── Test 7: drainEngineLogs returns empty when nothing pending ────────────────
void test_drain_empty()
{
    llmengine::LlamaCppEngine engine;
    auto logs = engine.drainEngineLogs();
    REQUIRE(logs.empty(), "drainEngineLogs initially empty");
}

}  // namespace

int main()
{
    std::fprintf(stdout, "=== LlamaCppEngine smoke tests ===\n");
#ifndef LLM_ENGINE_HAVE_LLAMA_CPP
    std::fprintf(stdout,
        "Note: compiled WITHOUT llama.cpp support "
        "(LLM_ENGINE_BUILD_LLAMA_CPP=OFF).\n"
        "Testing graceful-failure paths only.\n\n");
#else
    std::fprintf(stdout,
        "Note: compiled WITH llama.cpp support.\n"
        "Backend available; testing all paths.\n\n");
#endif

    test_construct();
    test_bad_path();
    test_attn_before_load();
    test_tokens_before_load();
    test_capabilities();
    test_unload_clears_view();
    test_drain_empty();

    if (g_fail == 0) {
        std::fprintf(stdout, "\nAll smoke tests PASSED.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d smoke test(s) FAILED.\n", g_fail);
    return 1;
}
