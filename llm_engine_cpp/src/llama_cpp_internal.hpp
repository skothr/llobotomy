#pragma once
// llama_cpp_internal.hpp — shared types for llama_cpp_engine.cpp and
// llama_cpp_capture.cpp.  Not part of the public API.

#ifdef LLM_ENGINE_HAVE_LLAMA_CPP

#include "llm_engine/capture.hpp"
#include "llm_engine/log.hpp"
#include "llm_engine/model.hpp"

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

// Forward-declare llama.cpp opaque types in the global namespace.
struct llama_context;
struct llama_model;

namespace llmengine {

// Shared capture context passed as cb_eval user_data during llama_decode.
struct LlamaCaptureCtx {
    int n_layers = 0;
    int n_heads  = 0;
    int n_seq    = 0;
    std::shared_ptr<CaptureBundle> bundle;

    // Set to true once the prompt prefill is complete and streaming
    // decodes begin.  When true, cb_eval skips writes to the per-layer
    // attention/residual maps so the prefill snapshot is preserved (a
    // single-token streaming decode would otherwise overwrite the
    // multi-position prompt attention with a degenerate 1×1 view).  The
    // streaming loop still updates token_strs/ids and logits per step.
    bool freeze_layer_writes = false;

    // Head ablations applied to this decode.  Snapshot of the engine's
    // intervention set at decode start (copied into cap_ctx by workerRun).
    // For each (layer, head) entry, cb_eval zeroes that head's row in the
    // kq_soft_max-{layer} tensor via ggml_backend_tensor_set write-back
    // BEFORE downstream ops consume it — actually modifying the forward
    // pass, not just the captured snapshot.
    std::set<std::pair<int, int>> ablated_heads;

    // Component ablations applied to this decode.  Each pair is
    // (layer, llama_tensor_basename) — the basename matches what cb_eval
    // sees BEFORE the "-{L}" suffix is appended (e.g. "attn_out",
    // "ffn_out").  Set is populated by LlamaCppEngine::setAblation
    // translating ComponentRef::component strings to llama.cpp's
    // canonical compute-graph names.
    std::set<std::pair<int, std::string>> ablated_components;

    // Sampler configuration snapshot.  llama_cpp_run_capture builds a
    // llama_sampler chain from this for the streaming loop.  Default
    // (Greedy) matches the engine's prior hard-coded behavior.
    SamplerConfig sampler_cfg;

    // Steering snapshot.  When steering.active && layer parses && the
    // direction vector matches d_model, cb_eval adds alpha * direction
    // to the target layer's l_out-{N} residual stream and writes back.
    // Affects every token position uniformly in this iteration.
    SteeringConfig steering;
    int            steering_layer = -1;  // pre-parsed from steering.layer
};

// Executes one forward pass with activation capture.  When
// `max_generation_tokens > 0`, also runs a greedy-sampling generation
// loop after the prompt prefill, publishing an updated bundle (with the
// new token appended and per-step capture state) via `publish_step`
// after each token.  `publish_step` may be null when no streaming UI
// subscriber is wired.  Defined in llama_cpp_capture.cpp.
using PublishStepFn = std::function<void(std::shared_ptr<const CaptureBundle>)>;

void llama_cpp_run_capture(
    llama_context*              ctx,
    llama_model*                lm,
    const std::vector<int32_t>& token_ids,
    int                         n_heads,
    LlamaCaptureCtx*            cap_ctx,
    std::vector<LogEntry>&      out_logs,
    int                         max_generation_tokens = 0,
    PublishStepFn               publish_step          = nullptr);

}  // namespace llmengine

#endif  // LLM_ENGINE_HAVE_LLAMA_CPP
