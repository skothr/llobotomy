// test_path_api.cpp — exhaustive coverage of ModelView::get(path).
//
// The canonical-path scheme is documented in model_view.hpp's top-of-
// file block + the SUPPORTED_ARCHITECTURES.md / README.  Consumers
// (serialisation, RPC, scripting, future ultra-tools) rely on every
// listed path resolving correctly.  This test enumerates them.
//
// Sections:
//   1. Every topology/* path returns the right variant arm.
//   2. Every capabilities/* path returns bool, value matches the
//      capabilities mirror.
//   3. Every tokenizer/* path returns string or bool.
//   4. provenance/* paths.
//   5. surgery/* paths (steering struct + ablation list + component list).
//   6. tensors/<name> path returns a TensorHandle.
//   7. Malformed paths fall through to monostate without crashing.

#include "llm_engine/model.hpp"
#include "llm_engine/model_view.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <typeinfo>
#include <variant>

using namespace llmengine;

namespace {

// Helper: assert that `v` holds the alternative T, return by VALUE.
// Returning by reference would dangle when the caller passes
// `v.get(path)` (a Value rvalue) — the temporary dies at end of the
// full expression, so binding the result to `const auto&` and using it
// later is UB.
template <class T>
T must_be(const ModelView::Value& v, const char* what) {
    if (!std::holds_alternative<T>(v)) {
        std::fprintf(stderr, "must_be<%s>(%s): variant index = %zu\n",
                     typeid(T).name(), what, v.index());
        std::abort();
    }
    return std::get<T>(v);
}

void test_topology_paths() {
    ModelView v;
    v.topology.name         = "test/m";
    v.topology.nLayers      = 12;
    v.topology.nHeads       = 16;
    v.topology.nKvHeads     = 4;
    v.topology.dModel       = 1024;
    v.topology.dHead        = 64;
    v.topology.dMlp         = 2048;
    v.topology.vocab        = 32000;
    v.topology.maxPos       = 4096;
    v.topology.ropeTheta    = 10000.0f;
    v.topology.totalParams  = 1'234'567'890;
    v.topology.chatTemplate = "<|user|>{prompt}</s>";
    v.topology.bosToken     = "<s>";
    v.topology.eosToken     = "</s>";

    assert(must_be<std::string>(v.get("topology/name"), "name")          == "test/m");
    assert(must_be<int>(v.get("topology/n_layers"), "n_layers")          == 12);
    assert(must_be<int>(v.get("topology/n_heads"), "n_heads")            == 16);
    assert(must_be<int>(v.get("topology/n_kv_heads"), "n_kv_heads")      == 4);
    assert(must_be<int>(v.get("topology/d_model"), "d_model")            == 1024);
    assert(must_be<int>(v.get("topology/d_head"), "d_head")              == 64);
    assert(must_be<int>(v.get("topology/d_mlp"), "d_mlp")                == 2048);
    assert(must_be<int>(v.get("topology/vocab"), "vocab")                == 32000);
    assert(must_be<int>(v.get("topology/max_pos"), "max_pos")            == 4096);
    assert(must_be<float>(v.get("topology/rope_theta"), "rope_theta")    == 10000.0f);
    assert(must_be<std::int64_t>(v.get("topology/total_params"), "tp")   == 1'234'567'890);
    assert(must_be<std::string>(v.get("topology/chat_template"), "tmpl") == "<|user|>{prompt}</s>");
    assert(must_be<std::string>(v.get("topology/bos_token"), "bos")      == "<s>");
    assert(must_be<std::string>(v.get("topology/eos_token"), "eos")      == "</s>");
}

void test_capabilities_paths() {
    ModelView v;
    v.capabilities.has_topology     = true;
    v.capabilities.has_tokenizer    = false;
    v.capabilities.has_state_dict   = true;
    v.capabilities.has_attention    = false;
    v.capabilities.has_residual     = false;
    v.capabilities.has_logit_lens   = false;
    v.capabilities.has_token_stream = false;
    v.capabilities.has_captures     = false;
    v.capabilities.has_intervention = false;
    v.capabilities.has_weight_deltas = false;
    v.capabilities.has_training     = false;

    assert(must_be<bool>(v.get("capabilities/has_topology"),       "t")  == true);
    assert(must_be<bool>(v.get("capabilities/has_tokenizer"),      "tk") == false);
    assert(must_be<bool>(v.get("capabilities/has_state_dict"),     "sd") == true);
    assert(must_be<bool>(v.get("capabilities/has_attention"),      "a")  == false);
    assert(must_be<bool>(v.get("capabilities/has_residual"),       "r")  == false);
    assert(must_be<bool>(v.get("capabilities/has_logit_lens"),     "ll") == false);
    assert(must_be<bool>(v.get("capabilities/has_token_stream"),   "ts") == false);
    assert(must_be<bool>(v.get("capabilities/has_captures"),       "c")  == false);
    assert(must_be<bool>(v.get("capabilities/has_intervention"),   "i")  == false);
    assert(must_be<bool>(v.get("capabilities/has_weight_deltas"),  "wd") == false);
    assert(must_be<bool>(v.get("capabilities/has_training"),       "tr") == false);
}

void test_tokenizer_paths() {
    ModelView v;
    v.tokenizer.bos_token     = "<s>";
    v.tokenizer.eos_token     = "</s>";
    v.tokenizer.pad_token     = "<pad>";
    v.tokenizer.chat_template = "tmpl";
    v.tokenizer.encode        = [](std::string_view) -> std::vector<TokenId> { return {}; };

    assert(must_be<std::string>(v.get("tokenizer/bos_token"),     "bos") == "<s>");
    assert(must_be<std::string>(v.get("tokenizer/eos_token"),     "eos") == "</s>");
    assert(must_be<std::string>(v.get("tokenizer/pad_token"),     "pad") == "<pad>");
    assert(must_be<std::string>(v.get("tokenizer/chat_template"), "ct")  == "tmpl");
    assert(must_be<bool>(v.get("tokenizer/has_encode"),           "he")  == true);
    assert(must_be<bool>(v.get("tokenizer/has_decode"),           "hd")  == false);
}

void test_provenance_paths() {
    ModelView v;
    v.provenance.path         = "/test/path.gguf";
    v.provenance.format       = "gguf";
    v.provenance.content_hash = "deadbeef";
    v.provenance.source_label = "test backend";

    assert(must_be<std::string>(v.get("provenance/path"),   "p") == "/test/path.gguf");
    assert(must_be<std::string>(v.get("provenance/format"), "f") == "gguf");
    assert(must_be<std::string>(v.get("provenance/hash"),   "h") == "deadbeef");
    assert(must_be<std::string>(v.get("provenance/source"), "s") == "test backend");
}

void test_surgery_paths() {
    ModelView v;
    v.surgery.steering.active = true;
    v.surgery.steering.alpha  = 0.5f;
    v.surgery.steering.layer  = "L08.resid_post";
    v.surgery.ablated_heads.push_back({3, 5});
    v.surgery.ablated_heads.push_back({0, 0});
    v.surgery.ablated_components.push_back({7, "mlp"});

    const auto& steering = must_be<SteeringConfig>(v.get("surgery/steering"), "steering");
    assert(steering.active);
    assert(steering.alpha == 0.5f);

    const auto& heads = must_be<std::vector<std::string>>(v.get("surgery/ablations"), "abl");
    assert(heads.size() == 2);
    assert(heads[0] == "blocks.3.attn.head.5");
    assert(heads[1] == "blocks.0.attn.head.0");

    const auto& comps = must_be<std::vector<std::string>>(v.get("surgery/components"), "cmp");
    assert(comps.size() == 1);
    assert(comps[0] == "blocks.7.mlp");
}

void test_tensors_path() {
    ModelView v;
    TensorHandle h;
    h.name  = "blocks.5.attn.W_K.weight";
    h.dtype = DType::F16;
    h.shape = {1024, 1024};
    v.tensors.insert(h);

    const auto& got = must_be<TensorHandle>(v.get("tensors/blocks.5.attn.W_K.weight"), "tensor");
    assert(got.name  == "blocks.5.attn.W_K.weight");
    assert(got.dtype == DType::F16);
    assert(got.shape.size() == 2);

    // Missing tensor → monostate.
    auto missing = v.get("tensors/no.such.tensor");
    assert(std::holds_alternative<std::monostate>(missing));
}

void test_malformed_paths_safe() {
    ModelView v;
    // Empty path
    assert(std::holds_alternative<std::monostate>(v.get("")));
    // Unknown root
    assert(std::holds_alternative<std::monostate>(v.get("nope")));
    assert(std::holds_alternative<std::monostate>(v.get("nope/whatever")));
    // Trailing slash on known root with no field
    assert(std::holds_alternative<std::monostate>(v.get("topology/")));
    // Known root, unknown field
    assert(std::holds_alternative<std::monostate>(v.get("topology/nonexistent")));
    assert(std::holds_alternative<std::monostate>(v.get("capabilities/has_warp_drive")));
    // Whitespace / control chars
    assert(std::holds_alternative<std::monostate>(v.get(" topology/n_layers ")));
}

}  // namespace

int main() {
    std::printf("topology...\n");     std::fflush(stdout); test_topology_paths();
    std::printf("capabilities...\n"); std::fflush(stdout); test_capabilities_paths();
    std::printf("tokenizer...\n");    std::fflush(stdout); test_tokenizer_paths();
    std::printf("provenance...\n");   std::fflush(stdout); test_provenance_paths();
    std::printf("surgery...\n");      std::fflush(stdout); test_surgery_paths();
    std::printf("tensors...\n");      std::fflush(stdout); test_tensors_path();
    std::printf("malformed...\n");    std::fflush(stdout); test_malformed_paths_safe();
    std::printf("all done\n");
    return 0;
}
