#pragma once
// llama_cpp_internal.hpp — shared types for llama_cpp_engine.cpp and
// llama_cpp_capture.cpp.  Not part of the public API.

#ifdef LLM_ENGINE_HAVE_LLAMA_CPP

#include "llm_engine/capture.hpp"
#include "llm_engine/log.hpp"

#include <memory>
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
};

// Executes one forward pass with activation capture.
// Defined in llama_cpp_capture.cpp.
void llama_cpp_run_capture(
    llama_context*              ctx,
    llama_model*                lm,
    const std::vector<int32_t>& token_ids,
    int                         n_heads,
    LlamaCaptureCtx*            cap_ctx,
    std::vector<LogEntry>&      out_logs);

}  // namespace llmengine

#endif  // LLM_ENGINE_HAVE_LLAMA_CPP
