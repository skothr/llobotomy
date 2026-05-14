// llama_cpp_capture.cpp — cb_eval callback wiring for LlamaCppEngine.
//
// Compiled only when LLM_ENGINE_HAVE_LLAMA_CPP=1.  Defines:
//   llama_cpp_run_capture() — called by the engine worker thread to execute
//                             one forward pass with activation capture.
//
// Capture strategy:
//   - Attention post-softmax tensors: tensor name "kq_soft_max-{L}" (or
//     "_ext-{L}" in newer versions).  Copy [seq, seq] per head into a
//     TensorHandle stored at attention[(L, H)].
//   - Residual stream: tensor name "l_out-{L}".  Copy [seq, d_model] into
//     a TensorHandle stored at residual_post[L].
//   - Output logits: captured post-decode via llama_get_logits() in the
//     engine worker (not here — this file is cb_eval only).
//
// Architecture coverage:
//   These tensor names are UNIVERSAL across llama.cpp's compute graphs,
//   not Llama-family-specific.  llama.cpp's cb_eval naming convention is
//   `ggml_format_name(cur, "%s-%d", name, il)` (llama-context.cpp:2200),
//   and every architecture file (gpt2, gptneox, gemma, qwen2, llama,
//   falcon, mistral, mixtral, phi, bloom, starcoder, codeshell, plamo,
//   arctic, glm4-moe, minicpm3, nemotron-h, ...) calls the same
//   `cb(cur, "l_out", il)` for the per-layer residual output.  The
//   shared `build_attn_inp_kv_unified` path in llama-graph.cpp emits the
//   same `cb(kq, "kq_soft_max", il)` regardless of architecture.
//
//   So no per-arch dispatch is needed for standard-attention models.
//   Architectures with custom non-softmax attention (mamba, RWKV-style
//   SSMs) won't emit kq_soft_max-* at all — they don't have softmax
//   attention.  Capture for those would need fundamentally different
//   logic (capturing state-space transitions instead of attention rows).
//
// The callback runs inside llama_decode() on the engine thread.  It must
// be fast and allocation-minimising.  We pre-copy to the pre-allocated
// LlamaCaptureCtx::bundle entries.
//
// Shape conventions:
//   kq_soft_max: [n_kv_heads, n_heads/n_kv_heads, seq, seq] or
//                [n_heads, seq, seq] — normalised to [head, seq_q, seq_k]
//                on copy.
//   l_out:       [d_model, seq, 1, 1] — copied per-row as [seq, d_model].

#ifdef LLM_ENGINE_HAVE_LLAMA_CPP

#include "llm_engine/capture.hpp"
#include "llm_engine/log.hpp"
#include "llm_engine/tensor_handle.hpp"
#include "llm_engine/tensor_source.hpp"
#include "llama_cpp_internal.hpp"

// llama.cpp SYSTEM headers.
#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llmengine {

// ─── Name parsing helpers ─────────────────────────────────────────────────────

// Try to parse "kq_soft_max-{N}" or "kq_soft_max_ext-{N}" → layer N.
// Returns -1 on mismatch.
static int parse_kq_softmax_layer(const char* name)
{
    // Prefixes to try, longest-first.
    static const char* prefixes[] = {
        "kq_soft_max_ext-",
        "kq_soft_max-",
        nullptr
    };
    for (const char** p = prefixes; *p; ++p) {
        std::string_view sv(name);
        std::string_view pfx(*p);
        if (sv.size() > pfx.size() && sv.substr(0, pfx.size()) == pfx) {
            std::string_view tail = sv.substr(pfx.size());
            // tail should be a decimal integer.
            int layer = 0;
            bool ok = !tail.empty();
            for (char c : tail) {
                if (c < '0' || c > '9') { ok = false; break; }
                layer = layer * 10 + (c - '0');
            }
            if (ok) return layer;
        }
    }
    return -1;
}

// Try to parse "l_out-{N}" → layer N (the residual stream after layer N).
// Llama-family tensor name in current llama.cpp.  Returns -1 on mismatch.
static int parse_residual_post_layer(const char* name)
{
    std::string_view sv(name);
    static const std::string_view pfx = "l_out-";
    if (sv.size() <= pfx.size() || sv.substr(0, pfx.size()) != pfx) return -1;
    std::string_view tail = sv.substr(pfx.size());
    int layer = 0;
    if (tail.empty()) return -1;
    for (char c : tail) {
        if (c < '0' || c > '9') return -1;
        layer = layer * 10 + (c - '0');
    }
    return layer;
}

// ─── cb_eval callback ─────────────────────────────────────────────────────────

// Signature required by ggml_backend_sched_eval_callback:
//   bool cb(ggml_tensor* t, bool ask, void* user_data)
//
// When ask==true:  return true if we want to observe this tensor.
// When ask==false: tensor data is ready; copy it.  Return true to continue,
//                  false to abort the graph (we never abort).
static bool capture_cb_eval(struct ggml_tensor* t, bool ask, void* user_data)
{
    auto* cap = static_cast<LlamaCaptureCtx*>(user_data);
    if (!cap || !cap->bundle) return true;

    // During streaming decodes (post-prefill), skip per-layer writes —
    // preserve the prefill snapshot so the UI keeps showing the full
    // prompt's attention pattern.  Logits + token_strs are still updated
    // per step by the streaming loop in llama_cpp_run_capture itself.
    if (cap->freeze_layer_writes) return true;

    const char* tname = ggml_get_name(t);
    if (!tname || tname[0] == '\0') return true;

    // Attention: kq_soft_max-{N}
    int attn_layer = parse_kq_softmax_layer(tname);
    // Residual stream: l_out-{N}
    int resid_layer = parse_residual_post_layer(tname);

    if (attn_layer < 0 && resid_layer < 0) return true;  // not a tensor we care about
    if (ask) return true;           // yes, we want to observe

    // Branch by tensor type.  Residual is the simpler shape: [d_model, seq, 1, 1].
    if (resid_layer >= 0) {
        if (t->type != GGML_TYPE_F32) return true;
        int64_t d_model = t->ne[0];
        int64_t seq     = t->ne[1];
        if (d_model <= 0 || seq <= 0) return true;
        std::size_t n_elem = static_cast<std::size_t>(d_model * seq);
        std::vector<float> data(n_elem);
        ggml_backend_tensor_get(t, data.data(), 0, n_elem * sizeof(float));

        auto src = InMemoryTensorSource::from_floats(std::move(data));
        TensorHandle th;
        th.source      = std::move(src);
        th.name        = std::string("residual_post[") + std::to_string(resid_layer) + "]";
        th.dtype       = DType::F32;
        th.shape       = {seq, d_model};   // canonical [seq, d_model]
        th.stride      = {};
        th.byte_offset = 0;
        th.byte_length = n_elem * sizeof(float);
        th.contiguous  = true;

        cap->bundle->residual_post[resid_layer] = std::move(th);
        return true;
    }

    // Otherwise: attention.
    int layer = attn_layer;

    // Tensor is ready.  Shape: ggml stores dimensions as ne[0..3].
    // For kq_soft_max the layout is [seq_k, seq_q, n_heads, ...] in ggml
    // (row-major: ne[0]=innermost).  We want [head, seq_q, seq_k].
    //
    // Specifically llama.cpp produces:
    //   ne[0] = kv_seq (key sequence length, == seq for a full prefill)
    //   ne[1] = q_seq  (query sequence length)
    //   ne[2] = n_heads (or n_kv_heads in GQA)
    //   ne[3] = 1
    //
    // We treat it as n_heads 2-D matrices of shape [q_seq, kv_seq].

    int64_t kv_seq  = t->ne[0];
    int64_t q_seq   = t->ne[1];
    int64_t n_heads = t->ne[2];
    if (kv_seq <= 0 || q_seq <= 0 || n_heads <= 0) return true;
    if (t->type != GGML_TYPE_F32) return true;  // only handle F32 output

    // Total bytes for all heads.
    std::size_t row_bytes   = static_cast<std::size_t>(kv_seq) * sizeof(float);
    std::size_t head_bytes  = static_cast<std::size_t>(q_seq)  * row_bytes;
    std::size_t total_bytes = static_cast<std::size_t>(n_heads) * head_bytes;

    // Allocate a single flat buffer and fetch from backend (handles CUDA).
    std::vector<float> flat(static_cast<std::size_t>(n_heads * q_seq * kv_seq));
    ggml_backend_tensor_get(t, flat.data(), 0, total_bytes);

    // ── Head ablation: zero out ablated heads' attention rows ────────────
    // For every (layer, head) in cap->ablated_heads matching this layer,
    // zero the head's [q_seq, kv_seq] slice in the CPU buffer, then write
    // back to the backend so downstream attention computations see zero
    // weights for those heads — this actually modifies the forward pass,
    // not just the captured snapshot.
    bool ablated_any = false;
    for (int64_t h = 0; h < n_heads; ++h) {
        if (cap->ablated_heads.count({layer, static_cast<int>(h)}) == 0) continue;
        std::size_t head_offset = static_cast<std::size_t>(h * q_seq * kv_seq);
        std::fill_n(flat.begin() + static_cast<std::ptrdiff_t>(head_offset),
                    static_cast<std::size_t>(q_seq * kv_seq), 0.0f);
        ablated_any = true;
    }
    if (ablated_any) {
        ggml_backend_tensor_set(t, flat.data(), 0, total_bytes);
    }

    // Split per head and store in CaptureBundle.
    for (int64_t h = 0; h < n_heads; ++h) {
        std::size_t head_offset = static_cast<std::size_t>(h * q_seq * kv_seq);
        std::vector<float> head_data(
            flat.begin() + static_cast<std::ptrdiff_t>(head_offset),
            flat.begin() + static_cast<std::ptrdiff_t>(head_offset + static_cast<std::size_t>(q_seq * kv_seq)));

        auto src = InMemoryTensorSource::from_floats(std::move(head_data));

        TensorHandle th;
        th.source      = std::move(src);
        th.name        = std::string("attention[") + std::to_string(layer)
                       + "," + std::to_string(h) + "]";
        th.dtype       = DType::F32;
        th.shape       = {q_seq, kv_seq};
        th.stride      = {};
        th.byte_offset = 0;
        th.byte_length = static_cast<std::size_t>(q_seq * kv_seq) * sizeof(float);
        th.contiguous  = true;

        cap->bundle->attention[{layer, static_cast<int>(h)}] = std::move(th);
    }

    return true;  // continue compute
}

// ─── llama_cpp_run_capture ────────────────────────────────────────────────────

// Called from the engine worker thread.  Executes one forward pass with the
// cb_eval callback registered to fill cap_ctx->bundle.
//
// The context's KV cache is cleared before the decode so each call is a
// fresh prefill (no incremental generation in this iteration).
void llama_cpp_run_capture(
    llama_context*              ctx,
    llama_model*                /*lm*/,
    const std::vector<int32_t>& token_ids,
    int                         /*n_heads*/,
    LlamaCaptureCtx*            cap_ctx,
    std::vector<LogEntry>&      out_logs,
    int                         max_generation_tokens,
    PublishStepFn               publish_step)
{
    // Clear KV cache from any prior run.
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) llama_memory_clear(mem, /*data=*/false);

    // Wire the eval callback for this decode.
    // llama_context_params.cb_eval is set at creation; we reach the sched
    // via llama_context_params which was already stored.  In this build the
    // cb is passed at context creation time, so we cannot change it per-call
    // on the public API without recreating the context.
    //
    // Workaround: we store the capture ctx pointer in a thread_local and
    // always register the callback at context-creation time pointing to that
    // thread_local.  See the CMake notes: we create the context once with
    // cb_eval=&dispatch_cb, and dispatch_cb reads the thread_local.
    //
    // For this first cut we sidestep this by recreating the context with a
    // fresh cb_eval per run — heavyweight but correct.  The context params
    // need the lm pointer; we grab it via llama_get_model.
    const llama_model* lm = llama_get_model(ctx);
    if (!lm) {
        LogEntry e; e.ts_ms=0; e.sev=Severity::Error; e.kind="llama_cpp";
        e.msg = "llama_cpp_run_capture: llama_get_model returned null";
        out_logs.push_back(std::move(e));
        return;
    }

    // ── Rebuild context with cb_eval wired ───────────────────────────────────
    //
    // We rebuild once per prompt.  For interpretability inspection this is
    // fine — the prompt changes rarely and the capture is what matters.
    // Future optimisation: store the context params and rebuild only when
    // the cb_eval user_data pointer changes.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx               = static_cast<uint32_t>(
        std::max(static_cast<int>(token_ids.size()) + 64, 512));
    cparams.n_batch             = static_cast<uint32_t>(token_ids.size());
    cparams.flash_attn_type     = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval             = capture_cb_eval;
    cparams.cb_eval_user_data   = cap_ctx;

    // llama_init_from_model takes a non-const llama_model* even though it
    // does not modify the model — cast is safe here.
    llama_context* tmp_ctx = llama_init_from_model(
        const_cast<llama_model*>(lm), cparams);
    if (!tmp_ctx) {
        LogEntry e; e.ts_ms=0; e.sev=Severity::Error; e.kind="llama_cpp";
        e.msg = "llama_cpp_run_capture: llama_init_from_model failed";
        out_logs.push_back(std::move(e));
        return;
    }

    // Build a batch from the token ids.
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token*>(token_ids.data()),
        static_cast<int32_t>(token_ids.size()));

    // Request logits for the last token only.
    // (batch.logits is nullptr from llama_batch_get_one → last token gets logits)

    int ret = llama_decode(tmp_ctx, batch);
    if (ret != 0) {
        LogEntry e; e.ts_ms=0; e.sev=Severity::Warn; e.kind="llama_cpp";
        e.msg = "llama_decode returned " + std::to_string(ret)
              + " (non-fatal; captures may be partial)";
        out_logs.push_back(std::move(e));
    }

    // ── Helper: capture logits from tmp_ctx into a bundle ────────────────
    const llama_vocab* vocab = llama_model_get_vocab(lm);
    const int n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;
    auto stash_logits = [&](CaptureBundle& b) {
        const float* lp = llama_get_logits(tmp_ctx);
        if (!lp || n_vocab <= 0) return;
        std::vector<float> data(lp, lp + n_vocab);
        auto src = InMemoryTensorSource::from_floats(std::move(data));
        TensorHandle th;
        th.source      = std::move(src);
        th.name        = "logits";
        th.dtype       = DType::F32;
        th.shape       = {1, n_vocab};
        th.byte_offset = 0;
        th.byte_length = static_cast<std::size_t>(n_vocab) * sizeof(float);
        th.contiguous  = true;
        b.logits = std::move(th);
    };

    // ── Post-decode logits capture (prompt prefill) ──────────────────────
    if (cap_ctx && cap_ctx->bundle && ret == 0) {
        stash_logits(*cap_ctx->bundle);
    }

    // ── Token-streaming generation loop (optional) ───────────────────────
    // After the prompt prefill, optionally sample greedily up to
    // `max_generation_tokens` more tokens.  Each step:
    //   1. sample next token from the bundle's current logits
    //   2. clone the bundle and append the new token
    //   3. swap cap_ctx->bundle to the new clone so cb_eval writes there
    //   4. decode the single new token (cb_eval fills the new bundle's
    //      attention/residual entries for the latest position)
    //   5. stash post-decode logits into the new bundle
    //   6. call publish_step(new_bundle) so the UI sees the streamed token
    //
    // Bundle copy per step keeps the previously-published bundle immutable
    // for any UI thread still holding a shared_ptr to it.
    if (ret == 0 && cap_ctx && cap_ctx->bundle && max_generation_tokens > 0 && vocab) {
        // Freeze per-layer writes from this point on — the prefill snapshot
        // is what the UI wants to inspect; streaming decodes only update
        // tokens + logits, not attention/residual maps.
        cap_ctx->freeze_layer_writes = true;

        // Build the sampler chain from cap_ctx->sampler_cfg.  Greedy is
        // a single-stage shortcut; Sampling builds the full chain
        // top_k → top_p → min_p → temperature → (mirostat or dist),
        // skipping stages at their identity values.
        const auto& cfg = cap_ctx->sampler_cfg;
        llama_sampler* smpl = nullptr;
        if (cfg.method == SamplerConfig::Method::Greedy) {
            smpl = llama_sampler_init_greedy();
        } else {
            auto sparams = llama_sampler_chain_default_params();
            smpl = llama_sampler_chain_init(sparams);
            if (smpl) {
                if (cfg.top_k > 0) {
                    llama_sampler_chain_add(smpl,
                        llama_sampler_init_top_k(cfg.top_k));
                }
                if (cfg.top_p < 1.0f) {
                    llama_sampler_chain_add(smpl,
                        llama_sampler_init_top_p(cfg.top_p, 1));
                }
                if (cfg.min_p > 0.0f) {
                    llama_sampler_chain_add(smpl,
                        llama_sampler_init_min_p(cfg.min_p, 1));
                }
                if (cfg.mirostat == 1) {
                    if (cfg.temperature != 1.0f) {
                        llama_sampler_chain_add(smpl,
                            llama_sampler_init_temp(cfg.temperature));
                    }
                    const int n_vocab = llama_vocab_n_tokens(vocab);
                    llama_sampler_chain_add(smpl,
                        llama_sampler_init_mirostat(
                            static_cast<int32_t>(n_vocab), cfg.seed,
                            cfg.mirostat_tau, cfg.mirostat_eta, 100));
                } else if (cfg.mirostat == 2) {
                    if (cfg.temperature != 1.0f) {
                        llama_sampler_chain_add(smpl,
                            llama_sampler_init_temp(cfg.temperature));
                    }
                    llama_sampler_chain_add(smpl,
                        llama_sampler_init_mirostat_v2(
                            cfg.seed, cfg.mirostat_tau, cfg.mirostat_eta));
                } else {
                    if (cfg.temperature != 1.0f) {
                        llama_sampler_chain_add(smpl,
                            llama_sampler_init_temp(cfg.temperature));
                    }
                    llama_sampler_chain_add(smpl,
                        llama_sampler_init_dist(cfg.seed));
                }
            }
        }
        if (!smpl) {
            LogEntry e; e.ts_ms=0; e.sev=Severity::Warn; e.kind="llama_cpp";
            e.msg = "sampler chain init returned null; streaming skipped";
            out_logs.push_back(std::move(e));
        } else {
            int generated = 0;
            while (generated < max_generation_tokens) {
                const llama_token next = llama_sampler_sample(smpl, tmp_ctx, -1);
                if (llama_vocab_is_eog(vocab, next)) break;

                auto new_bundle = std::make_shared<CaptureBundle>(*cap_ctx->bundle);
                new_bundle->token_ids.push_back(next);
                char tbuf[256] = {};
                int nn = llama_token_to_piece(vocab, next, tbuf, sizeof(tbuf) - 1, 0, true);
                new_bundle->token_strs.push_back(
                    (nn > 0) ? std::string(tbuf, static_cast<std::size_t>(nn)) : "");

                cap_ctx->bundle = new_bundle;

                std::vector<llama_token> one = {next};
                llama_batch nb = llama_batch_get_one(one.data(), 1);
                int rr = llama_decode(tmp_ctx, nb);
                if (rr != 0) {
                    LogEntry e; e.ts_ms=0; e.sev=Severity::Warn; e.kind="llama_cpp";
                    e.msg = "llama_decode (streaming) returned " + std::to_string(rr);
                    out_logs.push_back(std::move(e));
                    break;
                }

                stash_logits(*new_bundle);

                if (publish_step) {
                    publish_step(std::const_pointer_cast<const CaptureBundle>(new_bundle));
                }
                ++generated;
            }
            llama_sampler_free(smpl);
        }
    }

    llama_free(tmp_ctx);
}

}  // namespace llmengine

#endif  // LLM_ENGINE_HAVE_LLAMA_CPP
