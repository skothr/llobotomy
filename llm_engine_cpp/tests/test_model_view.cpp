// ModelView path-accessor tests.
//
// Covers the `Value get(path)` escape hatch — unknown paths return
// monostate, known scalars come back as the right variant alternative,
// tensor paths resolve into TensorHandle, etc.

#include "llm_engine/model_view.hpp"
#include "llm_engine/tensor_handle.hpp"

#include <cassert>
#include <string>
#include <variant>

using namespace llmengine;

int main() {
    ModelView v;
    v.provenance.path         = "/tmp/foo.gguf";
    v.provenance.format       = "gguf";
    v.provenance.source_label = "test";
    v.topology.name           = "tiny-test";
    v.topology.nLayers        = 22;
    v.topology.nHeads         = 32;
    v.topology.dModel         = 2048;
    v.topology.ropeTheta      = 10000.0f;
    v.topology.totalParams    = 1100048384;

    // String scalar
    {
        auto got = v.get("provenance/path");
        assert(std::holds_alternative<std::string>(got));
        assert(std::get<std::string>(got) == "/tmp/foo.gguf");
    }
    // Int scalar
    {
        auto got = v.get("topology/n_layers");
        assert(std::holds_alternative<int>(got));
        assert(std::get<int>(got) == 22);
    }
    // Float scalar
    {
        auto got = v.get("topology/rope_theta");
        assert(std::holds_alternative<float>(got));
        assert(std::get<float>(got) == 10000.0f);
    }
    // i64 scalar
    {
        auto got = v.get("topology/total_params");
        assert(std::holds_alternative<std::int64_t>(got));
        assert(std::get<std::int64_t>(got) == 1100048384);
    }
    // Tensor handle resolution
    {
        TensorHandle h;
        h.name  = "blocks.0.attn.W_Q.weight";
        h.dtype = DType::F16;
        h.shape = {2048, 2048};
        v.tensors.insert(h);

        auto got = v.get("tensors/blocks.0.attn.W_Q.weight");
        assert(std::holds_alternative<TensorHandle>(got));
        assert(std::get<TensorHandle>(got).shape.size() == 2);
    }
    // Unknown root
    {
        auto got = v.get("nope/whatever");
        assert(std::holds_alternative<std::monostate>(got));
    }
    // Unknown topology field
    {
        auto got = v.get("topology/not_a_field");
        assert(std::holds_alternative<std::monostate>(got));
    }

    // clear() resets everything to default.  After clear, scalar
    // getters return sentinels and tensor registry is empty.
    v.clear();
    {
        auto got = v.get("topology/n_layers");
        assert(std::holds_alternative<int>(got));
        // ModelInfo::nLayers defaults to kNoInt (-1) — the engine's "no
        // data" sentinel.
        assert(std::get<int>(got) == -1);
    }
    {
        auto got = v.get("tensors/blocks.0.attn.W_Q.weight");
        assert(std::holds_alternative<std::monostate>(got));
    }
    return 0;
}
