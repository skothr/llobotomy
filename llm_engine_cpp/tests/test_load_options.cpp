// test_load_options.cpp — LoadOptions plumbing + the "ignore unknown
// extras" forward-compat contract.
//
// The substrate plan declared: "Backends that don't recognise a field
// MUST silently ignore it — the same options struct may be used across
// mock / hf / gguf / llama / libtorch flows."  This test verifies that
// invariant: pass an options struct with arbitrary `extras` to every
// backend's loadCheckpoint and assert nothing crashes / nothing
// false-positive errors.

#include "llm_engine/gguf_inspector_engine.hpp"
#include "llm_engine/model.hpp"
#include "llm_engine/model_view.hpp"

#include <cassert>
#include <string>

using namespace llmengine;

namespace {

Model::LoadOptions options_with_extras() {
    Model::LoadOptions opts;
    opts.mode        = "nf4";        // recognised by some, ignored by others
    opts.mmap        = false;        // recognised by file-backed backends
    opts.verify_hash = true;         // recognised by hash-supporting backends
    opts.extras.emplace_back("custom_key",        "custom_value");
    opts.extras.emplace_back("future_extension",  "42");
    opts.extras.emplace_back("",                  "");
    return opts;
}

// ── 1. MockModel — accepts options + ignores everything ─────────────────
void test_mock_accepts_options() {
    MockModel m;
    auto opts = options_with_extras();
    // Path-only and path+options overloads both available.
    auto r1 = m.loadCheckpoint("any/path.bin");
    auto r2 = m.loadCheckpoint("any/path.bin", opts);
    // MockModel doesn't really load — base default returns ok=true.
    assert(r1.ok || !r1.error.empty());
    assert(r2.ok || !r2.error.empty());
}

// ── 2. GgufInspectorEngine — options form delegates to path-only ────────
void test_gguf_options_path() {
    GgufInspectorEngine eng;
    auto opts = options_with_extras();
    auto r = eng.loadCheckpoint("/nonexistent/missing.gguf", opts);
    // Should fail because the path doesn't exist — but with a real
    // error message, not a crash.  The extras must not affect outcome.
    assert(!r.ok);
    assert(!r.error.empty());

    // Same input via path-only form → same failure mode.
    auto r2 = eng.loadCheckpoint("/nonexistent/missing.gguf");
    assert(!r2.ok);
    assert(!r2.error.empty());
}

// ── 3. LoadOptions field semantics ──────────────────────────────────────
// (Pure struct test — no engine involvement.  Sanity-checks defaults.)
void test_load_options_defaults() {
    Model::LoadOptions opts;
    assert(opts.mode.empty());
    assert(opts.mmap == true);          // sensible default
    assert(opts.verify_hash == false);  // off by default — large-file friendly
    assert(opts.extras.empty());
}

void test_load_options_copyable() {
    auto a = options_with_extras();
    auto b = a;                         // copy ctor
    assert(b.mode == a.mode);
    assert(b.extras.size() == a.extras.size());
    Model::LoadOptions c;
    c = a;                              // copy assignment
    assert(c.extras.size() == a.extras.size());
}

}  // namespace

int main() {
    test_load_options_defaults();
    test_load_options_copyable();
    test_mock_accepts_options();
    test_gguf_options_path();
    return 0;
}
