#include "llm_engine/model_view.hpp"

#include <string>
#include <string_view>
#include <utility>

namespace llmengine {

namespace {

// Split path into [head, rest] on the first '/'. rest is empty when the
// path has no further segment.
std::pair<std::string_view, std::string_view> split1(std::string_view p) {
    const auto pos = p.find('/');
    if (pos == std::string_view::npos) return {p, {}};
    return {p.substr(0, pos), p.substr(pos + 1)};
}

}  // namespace

ModelView::Value ModelView::get(std::string_view path) const {
    auto [head, rest] = split1(path);

    if (head == "provenance") {
        auto [k, _] = split1(rest);
        if (k == "path")    return provenance.path;
        if (k == "format")  return provenance.format;
        if (k == "hash")    return provenance.content_hash;
        if (k == "source")  return provenance.source_label;
        return {};
    }

    if (head == "topology") {
        auto [k, _] = split1(rest);
        if (k == "name")          return topology.name;
        if (k == "n_layers")      return topology.nLayers;
        if (k == "n_heads")       return topology.nHeads;
        if (k == "n_kv_heads")    return topology.nKvHeads;
        if (k == "d_model")       return topology.dModel;
        if (k == "d_head")        return topology.dHead;
        if (k == "d_mlp")         return topology.dMlp;
        if (k == "vocab")         return topology.vocab;
        if (k == "max_pos")       return topology.maxPos;
        if (k == "rope_theta")    return topology.ropeTheta;
        if (k == "total_params")  return topology.totalParams;
        if (k == "chat_template") return topology.chatTemplate;
        if (k == "bos_token")     return topology.bosToken;
        if (k == "eos_token")     return topology.eosToken;
        return {};
    }

    if (head == "tensors") {
        // rest = "<tensor_name>" → return the handle.
        // Further extensions (#stats, #histogram) live behind DerivedCache;
        // wire those when a caller needs them. The path API stays minimal
        // — typed access is preferred for in-process code.
        if (const TensorHandle* h = tensors.find(rest)) {
            return *h;
        }
        return {};
    }

    if (head == "surgery") {
        auto [k, _] = split1(rest);
        if (k == "steering") return surgery.steering;
        return {};
    }

    return {};
}

void ModelView::clear() {
    provenance = {};
    topology   = {};
    tokenizer  = {};
    tensors    = {};
    captures.clear();
    current.store(nullptr);
    surgery    = {};
    derived.clear();
}

}  // namespace llmengine
