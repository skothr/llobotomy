#pragma once
//
// ModelView — the unified, in-memory data structure for an inspected
// model. One per session. Aggregates:
//
//   provenance   where the bytes came from
//   topology     n_layers / n_heads / dims / vocab
//   tokenizer    vocab + bos/eos + chat template
//   tensors      file-backed (or memory-backed) handles, lazy
//   captures     per-prompt forward-pass bundles (current = active)
//   surgery      ablations / steering / probes / weight deltas
//   derived      memoised analyses keyed by canonical path
//
// Backends mutate ModelView in place under their own threading model
// (see ENGINE_API.md §3). UI code reads the current capture via the
// atomic shared_ptr in `current` — lock-free on the hot path.
//
// The Value get(path) accessor exists as an escape hatch for
// serialisation, RPC, debugging. Typed field access is preferred for
// in-process code: view.topology.nLayers reads cleaner than
// view.get("topology/nLayers").

#include "llm_engine/capture.hpp"
#include "llm_engine/derived_cache.hpp"
#include "llm_engine/model.hpp"
#include "llm_engine/tensor_handle.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace llmengine {

// Where this view's data came from. Filled by the backend on
// loadCheckpoint; immutable afterward.
struct Provenance {
    std::string path;          // "/.../tinyllama.gguf" | "TinyLlama/TinyLlama-1.1B" (HF id)
    std::string format;        // "gguf" | "safetensors" | "torchscript" | "hf-proxy" | "mock"
    std::string content_hash;  // SHA256 of weight bytes if computed; "" otherwise
    std::string source_label;  // human-friendly: "HuggingFace via FastAPI"
};

// Tokenizer surface. encode/decode are std::function so the backend can
// plug in whatever runtime tokenizer it owns (HF tokenizers, sentencepiece,
// llama.cpp's, etc.) without leaking the implementation. Empty by default
// — backends that don't expose a tokenizer leave them unset; consumers
// must check operator bool() before calling.
struct TokenizerView {
    std::vector<std::string> id_to_token;     // lazy: empty until first need
    std::string              bos_token, eos_token, pad_token;
    std::string              chat_template;
    std::function<std::vector<TokenId>(std::string_view)> encode;
    std::function<std::string(TokenId)>                   decode;
};

// One ablation / probe attachment. Names use canonical paths so they
// can reference TensorRegistry entries directly.
struct AblationEntry {
    std::string name;          // "blocks.5.attn.head.3" or "blocks.7.mlp"
};
struct ProbeAttachment {
    std::string name;
    std::string location;      // "L08.resid_post"
    std::string kind;          // "linear" / "SAE" / ...
};

struct InterventionSet {
    std::vector<AblationEntry>   ablated_heads;
    std::vector<AblationEntry>   ablated_components;
    SteeringConfig               steering;
    std::vector<ProbeAttachment> probes;
    std::unordered_map<std::string, TensorHandle> weight_deltas;
};

struct ModelView {
    Provenance     provenance;
    ModelInfo      topology;          // alias kept; see model.hpp
    TokenizerView  tokenizer;
    TensorRegistry tensors;

    // Captures keyed by prompt_hash. Newest-active is duplicated as
    // `current` (atomic shared_ptr) for lock-free UI reads.
    std::unordered_map<std::string, std::shared_ptr<const CaptureBundle>> captures;
    std::atomic<std::shared_ptr<const CaptureBundle>> current;

    InterventionSet surgery;
    mutable DerivedCache derived;     // mutable: read-side compute updates the cache

    // Path-based escape hatch. Returns monostate on unknown / unsupported
    // paths. See model_view.cpp for the path grammar.
    using Value = std::variant<
        std::monostate,
        int, float, std::int64_t, std::string,
        TensorHandle, ResidualSummary, SteeringConfig,
        std::vector<LogitLensRow>, TensorStats
    >;
    Value get(std::string_view path) const;

    ModelView() = default;
    ModelView(const ModelView&)            = delete;
    ModelView& operator=(const ModelView&) = delete;
};

}  // namespace llmengine
