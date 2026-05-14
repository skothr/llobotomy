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
// Capabilities (post-load):
//   has_topology, has_tokenizer, has_captures, has_attention,
//   has_residual, has_logit_lens, has_state_dict, has_token_stream,
//   has_intervention = true.
//   has_intervention covers head ablation only — steering vector
//   addition is logged as a no-op (future work).
//   has_weight_deltas, has_training = false.

#include "llm_engine/model.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llmengine {

class LlamaCppEngine : public Model {
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

    // Intervention — head ablation modifies the next forward pass's
    // kq_soft_max-{L} tensors in cb_eval (zeroes out ablated heads via
    // ggml_backend_tensor_set write-back).  Component ablation and
    // steering vectors are recorded into view.surgery but not yet
    // applied to the forward pass (logged as a warn).
    void setAblation(const std::vector<std::string>& head_canonical,
                     const std::vector<std::string>& component_canonical) override;
    void setSteering(const SteeringConfig& cfg) override;
    void clearSteering() override;

    // Sampling control — full llama.cpp sampler chain via SamplerConfig
    // (top-k → top-p → min-p → temperature → mirostat, with optional
    // greedy override).  The next setActivePrompt call's streaming loop
    // builds a sampler chain from this config.  Default (greedy) matches
    // the engine's prior behavior.
    void setSamplerConfig(const SamplerConfig& cfg) override;

    // Max tokens to generate per prompt — clamps the streaming loop.
    // Pass <= 0 to disable streaming entirely (only prefill + capture).
    // Default (when never called) is 24 tokens.
    void setMaxGenerationTokens(int n) override;

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

    // Engine metrics — populates cuda_mem_{used,total}_GB via
    // ggml_backend_dev_memory on the first GPU backend (covers CUDA,
    // ROCm, Metal — anything ggml exposes as a GPU device).  device +
    // dtype strings reflect what llama.cpp loaded.  Other timing
    // fields remain sentinel until wired.
    EngineMetrics getEngineMetrics() override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace llmengine
