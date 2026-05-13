#pragma once
//
// LlamaCppEngine — Model implementation backed by an embedded llama.cpp
// runtime.  Links against the pre-built libllama.so at
//   /home/ai/ai-projects/llm/lib/llama.cpp/build/bin/
// when LLM_ENGINE_BUILD_LLAMA_CPP=ON (CMake option, default OFF).
//
// This file is always present in the source tree; guard blocks make every
// method a safe no-op when the feature is compiled out so consumers that
// don't need llama.cpp can include this header without linking errors.
//
// Thread model:
//   UI thread     — reads view().current.load() (lock-free atomic)
//   Engine thread — owns llama_context; runs llama_decode; the cb_eval
//                   callback fills an in-progress CaptureBundle on each
//                   tensor compute; atomic-stores the bundle after decode.
//
// Capabilities (this iteration):
//   has_topology, has_tokenizer, has_captures, has_attention = true
//   has_intervention, has_token_stream = false (future work)

#include "llm_engine/model.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llmengine {

class LlamaCppEngine : public MockModel {
public:
    LlamaCppEngine();
    ~LlamaCppEngine() override;

    LlamaCppEngine(const LlamaCppEngine&)            = delete;
    LlamaCppEngine& operator=(const LlamaCppEngine&) = delete;

    // Checkpoint lifecycle.
    // loadCheckpoint validates the path, loads the GGUF model via
    // llama_model_load_from_file, creates a context, and populates
    // view().topology / provenance.  Returns {ok=false, error=...} on any
    // failure without throwing.
    CheckpointResult loadCheckpoint(std::string_view path) override;
    CheckpointResult loadCheckpoint(std::string_view path,
                                    const LoadOptions& opts) override;
    void             unloadCheckpoint() override;

    // Prompt submission — copies the prompt string and wakes the engine
    // thread to run a forward pass + capture.
    void setActivePrompt(std::string_view prompt) override;

    // Attention pattern from the most-recent capture.  Returns {} until a
    // forward pass has completed.
    std::vector<std::vector<float>> getAttentionPattern(
        int layer, int head, int seqLen, HeadBias bias) override;

    std::vector<std::string> getCurrentTokens() override;

    // Per-layer residual stream summary derived from the most-recent
    // capture.  Computed lazily from the captured residual_post tensor.
    ResidualSummary getResidualSummary(int layer) override;

    // Top-k output logits from the most-recent capture's final-position
    // logits.  Returns {} when no capture is loaded.
    std::vector<LogitDist> getOutputLogits(int k) override;

    // Capability advertisement.
    Capabilities    getCapabilities() const override;

    // Unified view accessor.
    const ModelView& view() const override;

    // Log drain — pulls engine-thread log lines to the UI.
    std::vector<LogEntry> drainEngineLogs() override;

    // Progress (load is heavy; engine thread updates this).
    Progress getProgress() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace llmengine
