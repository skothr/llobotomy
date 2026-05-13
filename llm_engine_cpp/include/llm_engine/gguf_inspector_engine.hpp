#pragma once
//
// GgufInspectorEngine — native C++ backend that loads a .gguf file and
// exposes its tensor table + architecture metadata through the Model interface.
//
// Capability matrix (what the UI sees post-load):
//   has_topology    true   — populated from GGUF KV metadata
//   has_state_dict  true   — every tensor in the file is registered
//   has_tokenizer   false  — BOS/EOS strings extracted, encode/decode not wired
//   has_attention   false  — no inference; just weights
//   has_residual    false
//   has_logit_lens  false
//   has_token_stream false
//   has_captures    false
//   has_intervention false
//   has_weight_deltas false
//   has_training    false
//
// All the "no-inference" capabilities are false because this engine is a
// weight inspector: it parses a checkpoint and lets the UI browse the raw
// state-dict.  A future GgufInferenceEngine (wrapping llama.cpp) would set
// additional bits while inheriting the tensor-source / topology work here.
//
// Implementation pattern:
//   PIMPL with a ModelView member.  loadCheckpoint() is synchronous (the
//   header parse is fast; no heavy compute).  All "live" per-method getters
//   delegate to MockModel::method() (inherited) — they return mock data
//   when LLOB_USE_MOCK_DATA is on, sentinels otherwise.  Only view() and
//   getCapabilities() are overridden to return the real file-backed data.

#include "llm_engine/model.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace llmengine {

class GgufInspectorEngine final : public MockModel {
public:
    GgufInspectorEngine();
    ~GgufInspectorEngine() override;

    GgufInspectorEngine(const GgufInspectorEngine&)            = delete;
    GgufInspectorEngine& operator=(const GgufInspectorEngine&) = delete;

    // Synchronously parse the GGUF header, construct the TensorSource, and
    // populate the ModelView.  Returns {ok=false, error=...} on any parse
    // error without touching the previous view state — the engine stays in
    // whatever state it was before the failed load.
    CheckpointResult loadCheckpoint(std::string_view path) override;

    // Preferred form: accepts LoadOptions but currently only uses the path.
    // Ignores opts.mode / opts.verify_hash (no inference, no hash compute).
    CheckpointResult loadCheckpoint(std::string_view path,
                                    const LoadOptions& opts) override;

    // Reset the ModelView (clears topology, tensors, provenance).  The file
    // is closed / unmapped as a side-effect of releasing the GgufSource.
    void unloadCheckpoint() override;

    // Unified data accessor — returns the file-backed view.
    const ModelView& view() const override;

    // Capabilities reflect what this backend can satisfy.
    Capabilities getCapabilities() const override;

    // ModelInfo derived from the parsed header.
    ModelInfo getModelInfo() override;

    // Engine log — this backend pushes parse-time messages here so the
    // UI's log panel shows what was loaded (or why the load failed).
    std::vector<LogEntry> drainEngineLogs() override;

private:
    struct State;
    std::unique_ptr<State> m_state;
};

}  // namespace llmengine
