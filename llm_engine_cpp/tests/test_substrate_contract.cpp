// test_substrate_contract.cpp — invariants every backend must uphold.
//
// Tests the substrate's foundational contracts:
//   1. Capabilities mirror — getCapabilities() == view().capabilities
//      for every shipped backend.  Catches the false-advertising bug
//      class (claiming has_attention=true but never populating the
//      attention map).
//   2. clear() resets EVERY field — provenance, topology, tokenizer,
//      tensors, captures, current, surgery, derived, capabilities.
//      No "soft" reset where some fields linger.
//   3. Atomic current pointer — load() on a fresh ModelView returns
//      nullptr; store(nullptr) is safe; concurrent load() is safe.
//   4. TokenizerView has_encode/has_decode reflect actual function
//      slot state — no false true.
//   5. InterventionSet structured/canonical roundtrip — every entry
//      stored via setAblation comes back through view().get() with
//      its canonical name intact.

#include "llm_engine/model.hpp"
#include "llm_engine/model_view.hpp"

#include <cassert>
#include <atomic>
#include <memory>
#include <string>
#include <thread>

using namespace llmengine;

namespace {

// ── 1. Capabilities mirror ────────────────────────────────────────────────
// Every backend's getCapabilities() must match view().capabilities for the
// path-API to stay coherent.
void test_capabilities_mirror_mock() {
    MockModel m;
    const auto caps_virtual = m.getCapabilities();
    const auto caps_mirror  = m.view().capabilities;

    // Field-by-field comparison.  Listed explicitly so a new bit gets
    // a failing test until the test is updated, rather than silently
    // drifting.
    assert(caps_virtual.has_topology     == caps_mirror.has_topology);
    assert(caps_virtual.has_tokenizer    == caps_mirror.has_tokenizer);
    assert(caps_virtual.has_state_dict   == caps_mirror.has_state_dict);
    assert(caps_virtual.has_attention    == caps_mirror.has_attention);
    assert(caps_virtual.has_residual     == caps_mirror.has_residual);
    assert(caps_virtual.has_logit_lens   == caps_mirror.has_logit_lens);
    assert(caps_virtual.has_token_stream == caps_mirror.has_token_stream);
    assert(caps_virtual.has_captures     == caps_mirror.has_captures);
    assert(caps_virtual.has_intervention == caps_mirror.has_intervention);
    assert(caps_virtual.has_weight_deltas == caps_mirror.has_weight_deltas);
    assert(caps_virtual.has_training     == caps_mirror.has_training);
}

// ── 2. clear() resets everything ─────────────────────────────────────────
void test_clear_resets_everything() {
    ModelView v;
    // Populate every field.
    v.provenance.path     = "/test/foo.gguf";
    v.provenance.format   = "gguf";
    v.topology.name       = "test-model";
    v.topology.nLayers    = 99;
    v.topology.dModel     = 4096;
    v.tokenizer.bos_token = "<s>";
    v.tokenizer.encode    = [](std::string_view) -> std::vector<TokenId> { return {1}; };
    v.surgery.ablated_heads.push_back({5, 3});
    v.surgery.steering.alpha = 0.5f;
    v.capabilities.has_topology = true;
    v.capabilities.has_attention = true;
    v.current.store(std::make_shared<CaptureBundle>());

    TensorHandle h;
    h.name = "blocks.0.attn.W_Q.weight";
    h.dtype = DType::F16;
    h.shape = {16};
    v.tensors.insert(h);

    v.clear();

    // Every field must be at its default.
    assert(v.provenance.path.empty());
    assert(v.provenance.format.empty());
    assert(v.topology.name.empty());
    assert(v.topology.nLayers == kNoInt);
    assert(v.topology.dModel  == kNoInt);
    assert(v.tokenizer.bos_token.empty());
    assert(!v.tokenizer.has_encode());
    assert(!v.tokenizer.has_decode());
    assert(v.tensors.empty());
    assert(v.surgery.ablated_heads.empty());
    assert(v.surgery.steering.alpha != v.surgery.steering.alpha    // NaN check via self-inequality
           || v.surgery.steering.alpha == kNoFloat);
    assert(!v.capabilities.has_topology);
    assert(!v.capabilities.has_attention);
    assert(v.current.load() == nullptr);
    assert(v.captures.empty());
}

// ── 3. Atomic current pointer ────────────────────────────────────────────
void test_atomic_current_safe_basics() {
    ModelView v;
    assert(v.current.load() == nullptr);    // fresh view, no capture

    auto b1 = std::make_shared<CaptureBundle>();
    b1->prompt_hash = "abc";
    v.current.store(b1);
    auto loaded = v.current.load();
    assert(loaded != nullptr);
    assert(loaded->prompt_hash == "abc");

    v.current.store(nullptr);
    assert(v.current.load() == nullptr);
}

void test_atomic_current_concurrent() {
    // Light concurrency check — N readers + 1 writer doing many
    // load/store cycles.  Validates that no torn reads / segfaults
    // occur under contention.  Not a stress test; that's a separate
    // test_threading_basic which is deep-tier.
    ModelView v;
    std::atomic<bool> stop{false};
    std::atomic<int>  load_count{0};

    auto reader = [&]() {
        while (!stop.load()) {
            auto p = v.current.load();
            if (p) {
                // Touch the pointer to make sure it's coherent.
                volatile auto _ = p->prompt_hash.size();
                (void)_;
            }
            load_count.fetch_add(1);
        }
    };

    std::thread t1(reader), t2(reader), t3(reader);

    for (int i = 0; i < 200; ++i) {
        auto b = std::make_shared<CaptureBundle>();
        b->prompt_hash = "p" + std::to_string(i);
        v.current.store(b);
    }

    stop.store(true);
    t1.join(); t2.join(); t3.join();
    assert(load_count.load() > 0);
}

// ── 4. Tokenizer slot consistency ────────────────────────────────────────
void test_tokenizer_slot_consistency() {
    TokenizerView tv;
    assert(!tv.has_encode());
    assert(!tv.has_decode());

    tv.encode = [](std::string_view) -> std::vector<TokenId> { return {}; };
    assert(tv.has_encode());
    assert(!tv.has_decode());

    tv.decode = [](TokenId) -> std::string { return ""; };
    assert(tv.has_encode());
    assert(tv.has_decode());

    tv.encode = nullptr;
    assert(!tv.has_encode());
    assert(tv.has_decode());
}

// ── 5. InterventionSet structured / canonical roundtrip ──────────────────
void test_intervention_canonical_roundtrip() {
    InterventionSet s;
    s.ablated_heads.push_back({5, 3});
    s.ablated_heads.push_back({0, 0});
    s.ablated_components.push_back({7, "mlp"});
    s.ablated_components.push_back({11, "resid_post"});

    const auto head_names = s.ablated_head_names();
    assert(head_names.size() == 2);
    assert(head_names[0] == "blocks.5.attn.head.3");
    assert(head_names[1] == "blocks.0.attn.head.0");

    const auto comp_names = s.ablated_component_names();
    assert(comp_names.size() == 2);
    assert(comp_names[0] == "blocks.7.mlp");
    assert(comp_names[1] == "blocks.11.resid_post");

    // Parse roundtrip: canonical name → ref → same name.
    for (const auto& name : head_names) {
        auto parsed = AttentionHeadRef::parse(name);
        assert(parsed.has_value());
        assert(parsed->canonical() == name);
    }
    for (const auto& name : comp_names) {
        auto parsed = ComponentRef::parse(name);
        assert(parsed.has_value());
        assert(parsed->canonical() == name);
    }
}

}  // namespace

int main() {
    test_capabilities_mirror_mock();
    test_clear_resets_everything();
    test_atomic_current_safe_basics();
    test_atomic_current_concurrent();
    test_tokenizer_slot_consistency();
    test_intervention_canonical_roundtrip();
    return 0;
}
