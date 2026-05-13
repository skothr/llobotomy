#include "llm_engine/model_view.hpp"

#include <charconv>
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

// Parse an integer from a string_view.  Returns nullopt on any failure
// (empty, non-digit, overflow) — caller treats as parse failure.
std::optional<int> parse_int(std::string_view s) {
    if (s.empty()) return std::nullopt;
    int v = 0;
    auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || end != s.data() + s.size()) return std::nullopt;
    return v;
}

}  // namespace

// ─── AttentionHeadRef ────────────────────────────────────────────────────
// Canonical form: "blocks.{layer}.attn.head.{head}".  parse() accepts
// only that exact shape — strict round-trip with canonical().

std::string AttentionHeadRef::canonical() const {
    return "blocks." + std::to_string(layer) + ".attn.head." + std::to_string(head);
}

std::optional<AttentionHeadRef> AttentionHeadRef::parse(std::string_view s) {
    // Expected: "blocks.{N}.attn.head.{N}"
    constexpr std::string_view kPrefix = "blocks.";
    constexpr std::string_view kMid    = ".attn.head.";
    if (s.size() < kPrefix.size() + kMid.size() + 2) return std::nullopt;
    if (s.substr(0, kPrefix.size()) != kPrefix)      return std::nullopt;
    const auto mid_pos = s.find(kMid, kPrefix.size());
    if (mid_pos == std::string_view::npos)           return std::nullopt;

    const auto layer_str = s.substr(kPrefix.size(), mid_pos - kPrefix.size());
    const auto head_str  = s.substr(mid_pos + kMid.size());
    const auto layer = parse_int(layer_str);
    const auto head  = parse_int(head_str);
    if (!layer || !head) return std::nullopt;
    return AttentionHeadRef{*layer, *head};
}

// ─── ComponentRef ────────────────────────────────────────────────────────
// Canonical form: "blocks.{layer}.{component}" where component is a
// short tag from a known whitelist.  We don't enforce the whitelist at
// parse time — that's a backend concern — but we reject obviously
// malformed structures (missing tag, empty fields).

std::string ComponentRef::canonical() const {
    return "blocks." + std::to_string(layer) + "." + component;
}

std::optional<ComponentRef> ComponentRef::parse(std::string_view s) {
    constexpr std::string_view kPrefix = "blocks.";
    if (s.size() <= kPrefix.size())                 return std::nullopt;
    if (s.substr(0, kPrefix.size()) != kPrefix)     return std::nullopt;

    const auto dot_after_layer = s.find('.', kPrefix.size());
    if (dot_after_layer == std::string_view::npos)  return std::nullopt;
    const auto layer_str = s.substr(kPrefix.size(), dot_after_layer - kPrefix.size());
    const auto comp_str  = s.substr(dot_after_layer + 1);
    if (comp_str.empty())                            return std::nullopt;
    // Reject anything that's actually a head ref — those carry an
    // additional ".head.{N}" suffix and belong to AttentionHeadRef.
    if (comp_str.find('/') != std::string_view::npos) return std::nullopt;
    if (comp_str.find('.') != std::string_view::npos) return std::nullopt;
    const auto layer = parse_int(layer_str);
    if (!layer) return std::nullopt;
    return ComponentRef{*layer, std::string{comp_str}};
}

// ─── InterventionSet name accessors ──────────────────────────────────────

std::vector<std::string> InterventionSet::ablated_head_names() const {
    std::vector<std::string> out;
    out.reserve(ablated_heads.size());
    for (const auto& h : ablated_heads) out.push_back(h.canonical());
    return out;
}

std::vector<std::string> InterventionSet::ablated_component_names() const {
    std::vector<std::string> out;
    out.reserve(ablated_components.size());
    for (const auto& c : ablated_components) out.push_back(c.canonical());
    return out;
}

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
        if (k == "steering")   return surgery.steering;
        if (k == "ablations")  return surgery.ablated_head_names();
        if (k == "components") return surgery.ablated_component_names();
        return {};
    }

    if (head == "capabilities") {
        auto [k, _] = split1(rest);
        if (k == "has_topology")       return capabilities.has_topology;
        if (k == "has_tokenizer")      return capabilities.has_tokenizer;
        if (k == "has_state_dict")     return capabilities.has_state_dict;
        if (k == "has_attention")      return capabilities.has_attention;
        if (k == "has_residual")       return capabilities.has_residual;
        if (k == "has_logit_lens")     return capabilities.has_logit_lens;
        if (k == "has_token_stream")   return capabilities.has_token_stream;
        if (k == "has_captures")       return capabilities.has_captures;
        if (k == "has_intervention")   return capabilities.has_intervention;
        if (k == "has_weight_deltas")  return capabilities.has_weight_deltas;
        if (k == "has_training")       return capabilities.has_training;
        return {};
    }

    if (head == "tokenizer") {
        auto [k, _] = split1(rest);
        if (k == "bos_token")     return tokenizer.bos_token;
        if (k == "eos_token")     return tokenizer.eos_token;
        if (k == "pad_token")     return tokenizer.pad_token;
        if (k == "chat_template") return tokenizer.chat_template;
        if (k == "has_encode")    return tokenizer.has_encode();
        if (k == "has_decode")    return tokenizer.has_decode();
        return {};
    }

    return {};
}

void ModelView::clear() {
    provenance   = {};
    topology     = {};
    tokenizer    = {};
    tensors      = {};
    captures.clear();
    current.store(nullptr);
    surgery      = {};
    derived.clear();
    capabilities = {};
}

}  // namespace llmengine
