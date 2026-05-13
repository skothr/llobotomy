#pragma once
//
// CaptureBundle — everything one forward pass produced.
//
// All large fields are TensorHandles backed by an InMemoryTensorSource
// the bundle owns, so consumers (UI, derived analyses) use the same
// read_slice API as for static weights. The sparse maps let backends
// populate only what was actually requested — UI asks for a single
// (layer, head) attention matrix; the backend can lazily fill it.
//
// CaptureBundles are shared between the engine thread (writes) and
// the UI thread (reads) via the std::atomic<std::shared_ptr<const
// CaptureBundle>> in ModelView. Once a bundle is "current" it is
// effectively immutable — backends that need to extend it allocate a
// new bundle and atomic-swap.

#include "llm_engine/tensor_handle.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llmengine {

using TokenId = std::int32_t;

// Forward declaration from model.hpp — full definition there.
struct ResidualSummary;

struct CaptureBundle {
    // Identity key. Backends set this to a stable hash of the prompt +
    // any intervention state so multiple captures (base run vs ablated
    // run) coexist in the same ModelView for A/B comparison.
    std::string prompt_hash;

    std::vector<TokenId>     token_ids;
    std::vector<std::string> token_strs;

    // Sparse maps keyed by (layer, head) or layer. Backends populate
    // entries on demand — UI requests, backend fills, next frame reads.
    std::map<std::pair<int, int>, TensorHandle> attention;     // (L,H) → [seq, seq]
    std::map<int, TensorHandle>                 residual_pre;  // L → [seq, d_model]
    std::map<int, TensorHandle>                 residual_post; // L → [seq, d_model]
    std::map<int, TensorHandle>                 mlp_post;      // L → [seq, d_mlp]
    std::map<std::pair<int, int>, TensorHandle> q, k, v;       // (L,H) → [seq, d_head]
    std::optional<TensorHandle>                 logits;        // [seq, vocab]

    // Cheap reductions a backend can pre-compute at capture time. Keyed
    // by layer; UI's residual_flow panel reads these per frame.
    // (Lives on the bundle, not in DerivedCache, because the reductions
    //  are part of the capture's content rather than a derived analysis
    //  over it — they're cheap at capture time and there's no caller
    //  who wants them without also wanting the underlying handles.)
    // Stored in a unique_ptr<map<…>> indirection so this header can
    // declare CaptureBundle without including the (large) model.hpp.
    // See capture.cpp for the inline get/put helpers.
    std::map<int, std::shared_ptr<ResidualSummary>> residual_summary;

    // Snapshot version for the double-buffer pattern. The engine
    // increments this with each new bundle so UI code can detect "the
    // current capture changed" without comparing pointers.
    std::uint64_t version = 0;
};

}  // namespace llmengine
