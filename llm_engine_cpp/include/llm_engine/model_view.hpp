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
//
// ─── Threading contract ──────────────────────────────────────────────
//
// ModelView has three lifecycle phases.  Each field below lists which
// phase it can be written in and what concurrent reads are safe.
//
//   1. BUILD phase — runs inside Model::loadCheckpoint on the engine
//      thread.  Single writer, no readers (UI hasn't been told a
//      checkpoint is ready).  Backends populate:
//        provenance        — set once, frozen for the session
//        topology          — set once, frozen
//        tokenizer         — set once, frozen
//        tensors           — registry filled; no concurrent reads
//        surgery           — typically default-constructed; backend may
//                            seed defaults
//
//   2. STEADY-STATE phase — runs until unloadCheckpoint.  UI reads
//      every frame; the engine writes only to the fields below:
//        captures map      — writes through `current` (atomic swap);
//                            map itself is single-writer.  UI reads
//                            via `current.load()` only.
//        current           — std::atomic<std::shared_ptr<...>>.  Safe
//                            for concurrent UI reads + engine writes.
//        surgery           — backend mutates under its own internal
//                            mutex; UI snapshots before reading.  Use
//                            setAblation / setSteering, never mutate
//                            directly from outside the backend.
//        derived           — internally synchronised; safe at frame rate.
//
//      The static fields (provenance, topology, tokenizer, tensors)
//      are READ-ONLY in steady state.  Mutating them is a programming
//      error.
//
//   3. UNLOAD phase — runs inside Model::unloadCheckpoint.  ModelView::clear()
//      resets every field; backends MUST quiesce all worker threads
//      before calling it.  Callers must hold no shared_ptrs into the
//      view's captures during clear() (the captures map is destroyed).

#include "llm_engine/capture.hpp"
#include "llm_engine/derived_cache.hpp"
#include "llm_engine/model.hpp"
#include "llm_engine/tensor_handle.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
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
// must check has_encode() / has_decode() before calling.
struct TokenizerView {
    std::vector<std::string> id_to_token;     // lazy: empty until first need
    std::string              bos_token, eos_token, pad_token;
    std::string              chat_template;
    std::function<std::vector<TokenId>(std::string_view)> encode;
    std::function<std::string(TokenId)>                   decode;

    bool has_encode() const { return static_cast<bool>(encode); }
    bool has_decode() const { return static_cast<bool>(decode); }
};

// ─── Structured intervention targets ─────────────────────────────────
//
// The substrate identifies what's being intervened on with strongly-
// typed refs instead of bare strings.  Each ref carries layer/head/
// component as integers + canonical()/parse() bridges to a stable
// canonical-path string scheme that path APIs (Value get) and storage
// (DerivedCache, serialisation) can share unambiguously.
//
// Path scheme:
//   AttentionHeadRef → "blocks.{layer}.attn.head.{head}"
//   ComponentRef     → "blocks.{layer}.{component}"
//                      component ∈ {"attn", "mlp", "ln1", "ln2",
//                                   "resid_pre", "resid_post", "mlp_post"}
//
// parse() returns std::nullopt on malformed input; canonical() always
// produces a path that round-trips through parse().
struct AttentionHeadRef {
    int layer = -1;
    int head  = -1;

    std::string canonical() const;
    static std::optional<AttentionHeadRef> parse(std::string_view canonical);

    bool operator==(const AttentionHeadRef&) const = default;
};

struct ComponentRef {
    int         layer = -1;
    std::string component;        // "attn" | "mlp" | "ln1" | ...

    std::string canonical() const;
    static std::optional<ComponentRef> parse(std::string_view canonical);

    bool operator==(const ComponentRef&) const = default;
};

struct ProbeAttachment {
    std::string  name;            // probe's own name, e.g. "linear/refusal_dir"
    ComponentRef location;        // where it's attached
    std::string  kind;            // "linear" | "SAE" | "logit" / ...
};

struct InterventionSet {
    std::vector<AttentionHeadRef> ablated_heads;
    std::vector<ComponentRef>     ablated_components;
    SteeringConfig                steering;
    std::vector<ProbeAttachment>  probes;
    std::unordered_map<std::string, TensorHandle> weight_deltas;

    // Convenience accessors for path-API consumers — return the
    // canonical-name list (typed callers prefer the structured fields
    // above).  Each call materialises a fresh vector; cheap enough for
    // the path API's escape-hatch role.
    std::vector<std::string> ablated_head_names() const;
    std::vector<std::string> ablated_component_names() const;
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
    //
    // Alternatives are append-only per the substrate's ABI policy.
    // Adding a type is forward-compat for consumers (they handle the
    // new variant arm or fall into monostate via std::visit's default);
    // re-ordering is a major-version break.
    using Value = std::variant<
        std::monostate,
        bool, int, float, std::int64_t, std::string,
        TensorHandle, ResidualSummary, SteeringConfig,
        std::vector<std::string>,
        std::vector<LogitLensRow>, TensorStats
    >;
    Value get(std::string_view path) const;

    // Capabilities snapshot — set by the owning Model.  Wired by the
    // Model::view() implementation post-load so the path API can resolve
    // `capabilities/has_X`.  Default-constructed (all false) until then.
    Model::Capabilities capabilities;

    // Reset every field to default.  Called by Model::unloadCheckpoint
    // to guarantee a fresh session never inherits state from a prior
    // one (ablations from model A clinging to model B's tensors, etc.).
    // Caller MUST quiesce all engine-side worker threads first.
    void clear();

    ModelView() = default;
    ModelView(const ModelView&)            = delete;
    ModelView& operator=(const ModelView&) = delete;
};

}  // namespace llmengine
