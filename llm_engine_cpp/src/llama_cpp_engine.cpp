// llama_cpp_engine.cpp — LlamaCppEngine implementation.
//
// Compiled only when LLM_ENGINE_HAVE_LLAMA_CPP=1 (set by CMake when
// LLM_ENGINE_BUILD_LLAMA_CPP=ON).  The header always compiles; this
// translation unit links against libllama.so.

#include "llm_engine/llama_cpp_engine.hpp"
#include "llm_engine/capture.hpp"
#include "llm_engine/log.hpp"
#include "llm_engine/model_view.hpp"
#include "llm_engine/tensor_source.hpp"
#include "llama_cpp_internal.hpp"

// Standard headers used by both #ifdef branches (cmath / algorithm by
// the Wave D extension methods like getResidualSummary + getOutputLogits;
// these methods are guarded individually below).  Including unconditionally
// keeps the headers available wherever they're needed.
#include <algorithm>
#include <cmath>
#include <utility>

#ifdef LLM_ENGINE_HAVE_LLAMA_CPP

// llama.cpp public headers — SYSTEM include so their warnings stay quiet.
#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>

#endif  // LLM_ENGINE_HAVE_LLAMA_CPP

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace llmengine {

// ─── Impl ────────────────────────────────────────────────────────────────────

struct LlamaCppEngine::Impl {
    // ── Guarded state ──────────────────────────────────────────────────
    mutable std::mutex mu;
    std::vector<LogEntry> pending_logs;

    ModelView view;     // provenance + topology + current capture

    // ── llama.cpp handles ──────────────────────────────────────────────
#ifdef LLM_ENGINE_HAVE_LLAMA_CPP
    llama_model*   lm  = nullptr;
    llama_context* ctx = nullptr;
#endif

    // ── Worker thread ──────────────────────────────────────────────────
    std::thread             worker;
    std::atomic<bool>       stop_flag{false};
    std::condition_variable wake_cv;

    // Prompt queue: one slot (newest wins).
    std::string  pending_prompt;    // "" = nothing pending
    bool         has_pending = false;

    // Progress for heavy load.
    mutable std::mutex prog_mu;
    Model::Progress    progress;

    // ── Lifecycle ──────────────────────────────────────────────────────
    Impl() = default;

    ~Impl() {
        stopWorker();
#ifdef LLM_ENGINE_HAVE_LLAMA_CPP
        freeHandles();
#endif
    }

    void stopWorker() {
        stop_flag.store(true);
        wake_cv.notify_all();
        if (worker.joinable()) worker.join();
    }

#ifdef LLM_ENGINE_HAVE_LLAMA_CPP
    void freeHandles() {
        if (ctx) { llama_free(ctx);            ctx = nullptr; }
        if (lm)  { llama_model_free(lm);       lm  = nullptr; }
    }
#endif

    void pushLog(Severity sev, std::string msg) {
        LogEntry e;
        e.ts_ms = 0;
        e.sev   = sev;
        e.kind  = "llama_cpp";
        e.msg   = std::move(msg);
        std::lock_guard<std::mutex> lk(mu);
        pending_logs.push_back(std::move(e));
    }

    void setProgress(const Model::Progress& p) {
        std::lock_guard<std::mutex> lk(prog_mu);
        progress = p;
    }

    // ── Worker entry point ─────────────────────────────────────────────
    void workerRun();
};

// ─── LlamaCppEngine public API ────────────────────────────────────────────────

LlamaCppEngine::LlamaCppEngine()
    : m_impl(std::make_unique<Impl>())
{
#ifdef LLM_ENGINE_HAVE_LLAMA_CPP
    llama_backend_init();
#endif
}

LlamaCppEngine::~LlamaCppEngine() = default;

// ── loadCheckpoint ────────────────────────────────────────────────────────────

Model::CheckpointResult
LlamaCppEngine::loadCheckpoint(std::string_view path)
{
    return loadCheckpoint(path, {});
}

Model::CheckpointResult
LlamaCppEngine::loadCheckpoint(std::string_view path,
                               const LoadOptions& /*opts*/)
{
#ifndef LLM_ENGINE_HAVE_LLAMA_CPP
    (void)path;
    return {false, "LlamaCppEngine: built without llama.cpp support"};
#else
    // Stop any running worker, free old handles.
    m_impl->stopWorker();
    m_impl->stop_flag.store(false);   // reset for next worker

    {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        m_impl->freeHandles();
        m_impl->view.clear();
        m_impl->pending_logs.clear();
        m_impl->has_pending = false;
        m_impl->pending_prompt.clear();
    }

    m_impl->setProgress({"load", 0, 0});

    // Load model.
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;  // offload all to GPU when available

    std::string path_str(path);

    llama_model* lm = llama_model_load_from_file(path_str.c_str(), mparams);
    if (!lm) {
        m_impl->setProgress({});
        return {false, "llama_model_load_from_file failed for: " + path_str};
    }

    // Create context with cb_eval wired.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 2048;
    cparams.n_batch    = 512;
    // Flash attention off: we need to intercept intermediate tensors.
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // cb_eval: registered per-decode (to pass the capture ctx); set null here.
    cparams.cb_eval           = nullptr;
    cparams.cb_eval_user_data = nullptr;

    llama_context* lctx = llama_init_from_model(lm, cparams);
    if (!lctx) {
        llama_model_free(lm);
        m_impl->setProgress({});
        return {false, "llama_init_from_model failed"};
    }

    // Stash handles.
    {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        m_impl->lm  = lm;
        m_impl->ctx = lctx;
    }

    // Populate topology.
    const llama_vocab* vocab = llama_model_get_vocab(lm);

    ModelInfo info;
    info.name    = path_str;
    info.nLayers = llama_model_n_layer(lm);
    info.nHeads  = llama_model_n_head(lm);
    info.nKvHeads= llama_model_n_head_kv(lm);
    info.dModel  = llama_model_n_embd(lm);
    info.dHead   = (info.nHeads > 0) ? (info.dModel / info.nHeads) : kNoInt;
    info.vocab   = llama_vocab_n_tokens(vocab);
    info.maxPos  = llama_model_n_ctx_train(lm);

    // Chat template (may be nullptr).
    const char* tmpl = llama_model_chat_template(lm, nullptr);
    if (tmpl) info.chatTemplate = tmpl;

    {
        // BOS/EOS tokens.
        llama_token bos_id = llama_vocab_bos(vocab);
        llama_token eos_id = llama_vocab_eos(vocab);
        if (bos_id >= 0) {
            const char* t = llama_vocab_get_text(vocab, bos_id);
            if (t) info.bosToken = t;
        }
        if (eos_id >= 0) {
            const char* t = llama_vocab_get_text(vocab, eos_id);
            if (t) info.eosToken = t;
        }
    }

    {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        m_impl->view.topology = info;

        m_impl->view.provenance.path         = path_str;
        m_impl->view.provenance.format       = "llama_cpp";
        m_impl->view.provenance.source_label = "llama.cpp in-process";
        m_impl->view.provenance.content_hash = "";

        // Wire tokenizer encode/decode via llama.cpp vocab.
        m_impl->view.tokenizer.encode = [lm, vocab](std::string_view text) {
            std::vector<llama_token> toks(text.size() + 16);
            int n = llama_tokenize(vocab,
                text.data(), static_cast<int32_t>(text.size()),
                toks.data(), static_cast<int32_t>(toks.size()),
                /*add_special=*/true, /*parse_special=*/false);
            if (n < 0) {
                toks.resize(-n);
                n = llama_tokenize(vocab,
                    text.data(), static_cast<int32_t>(text.size()),
                    toks.data(), static_cast<int32_t>(toks.size()),
                    true, false);
            }
            if (n < 0) return std::vector<TokenId>{};
            return std::vector<TokenId>(toks.begin(), toks.begin() + n);
        };
        m_impl->view.tokenizer.decode = [vocab](TokenId id) {
            char buf[256] = {};
            int n = llama_token_to_piece(vocab, id, buf, sizeof(buf) - 1,
                                         /*lstrip=*/0, /*special=*/true);
            if (n <= 0) return std::string{};
            return std::string(buf, static_cast<std::size_t>(n));
        };

        m_impl->view.capabilities = getCapabilities();
    }

    m_impl->setProgress({});

    // Start engine worker.
    m_impl->worker = std::thread([this] { m_impl->workerRun(); });

    return {true, ""};
#endif  // LLM_ENGINE_HAVE_LLAMA_CPP
}

// ── unloadCheckpoint ──────────────────────────────────────────────────────────

void LlamaCppEngine::unloadCheckpoint()
{
    m_impl->stopWorker();
    m_impl->stop_flag.store(false);
#ifdef LLM_ENGINE_HAVE_LLAMA_CPP
    {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        m_impl->freeHandles();
        m_impl->has_pending = false;
        m_impl->pending_prompt.clear();
    }
    m_impl->view.clear();
#endif
}

// ── setActivePrompt ───────────────────────────────────────────────────────────

void LlamaCppEngine::setActivePrompt(std::string_view prompt)
{
    {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        m_impl->pending_prompt = std::string(prompt);
        m_impl->has_pending    = true;
    }
    m_impl->wake_cv.notify_one();
}

// ── getAttentionPattern ───────────────────────────────────────────────────────

std::vector<std::vector<float>>
LlamaCppEngine::getAttentionPattern(int layer, int head,
                                    int /*seqLen*/, HeadBias /*bias*/)
{
    auto bundle = m_impl->view.current.load();
    if (!bundle) return {};

    auto it = bundle->attention.find({layer, head});
    if (it == bundle->attention.end()) return {};

    const TensorHandle& h = it->second;
    int64_t seq = (h.shape.size() >= 2) ? h.shape[0] : 0;
    if (seq <= 0) return {};

    return h.read_slice_2d(0, static_cast<std::size_t>(seq),
                           0, static_cast<std::size_t>(seq));
}

// ── getCurrentTokens ──────────────────────────────────────────────────────────

std::vector<std::string> LlamaCppEngine::getCurrentTokens()
{
    auto bundle = m_impl->view.current.load();
    if (!bundle) return {};
    return bundle->token_strs;
}

// ── getResidualSummary ────────────────────────────────────────────────────────
//
// Reads the captured residual_post[layer] tensor and computes per-layer
// scalar summaries (norms, kurtosis).  cos_prev would compare to the
// previous layer's residual — leave as sentinel until both layers'
// captures are available.

ResidualSummary LlamaCppEngine::getResidualSummary(int layer)
{
    ResidualSummary s;
    auto bundle = m_impl->view.current.load();
    if (!bundle) return s;
    auto it = bundle->residual_post.find(layer);
    if (it == bundle->residual_post.end()) return s;

    const TensorHandle& h = it->second;
    // Shape [seq, d_model] — flatten and compute scalars on the last token
    // (most informative for "what does the residual look like right now").
    if (h.shape.size() < 2) return s;
    const std::size_t seq     = static_cast<std::size_t>(h.shape[0]);
    const std::size_t d_model = static_cast<std::size_t>(h.shape[1]);
    if (seq == 0 || d_model == 0) return s;

    // Last-token slice.
    auto row = h.read_slice((seq - 1) * d_model, d_model);
    if (row.size() != d_model) return s;

    // L2 norm.
    double sq = 0.0;
    for (float v : row) sq += static_cast<double>(v) * v;
    s.resid_norm = static_cast<float>(std::sqrt(sq));

    // Excess kurtosis (Fisher).  E[(x - μ)^4] / σ^4 - 3.
    double mean = 0.0;
    for (float v : row) mean += v;
    mean /= static_cast<double>(d_model);
    double m2 = 0.0, m4 = 0.0;
    for (float v : row) {
        double d = static_cast<double>(v) - mean;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    m2 /= static_cast<double>(d_model);
    m4 /= static_cast<double>(d_model);
    if (m2 > 0.0) {
        s.kurtosis = static_cast<float>(m4 / (m2 * m2) - 3.0);
    }

    // attn_out_norm / mlp_out_norm — would need separate captures of those
    // intermediate tensors.  Leave as sentinel for this iteration.
    return s;
}

// ── getOutputLogits ───────────────────────────────────────────────────────────
//
// Top-k logits from the final position of the most-recent capture.
// Applies softmax to the raw logits so the returned probs sum to 1 (the
// UI expects probabilities, not raw logits).  Token strings come from
// llama_token_to_piece on the cached model + tokenizer.

std::vector<LogitDist> LlamaCppEngine::getOutputLogits(int k)
{
    auto bundle = m_impl->view.current.load();
    if (!bundle || !bundle->logits.has_value()) return {};

    const TensorHandle& h = *bundle->logits;
    const std::size_t n_vocab =
        h.shape.size() >= 2 ? static_cast<std::size_t>(h.shape.back()) : 0;
    if (n_vocab == 0) return {};

    auto raw = h.read_slice(0, n_vocab);
    if (raw.size() != n_vocab) return {};

    // Softmax with max-subtract for numerical stability.
    float mx = raw[0];
    for (float v : raw) if (v > mx) mx = v;
    std::vector<float> probs(n_vocab);
    double sum = 0.0;
    for (std::size_t i = 0; i < n_vocab; ++i) {
        probs[i] = std::exp(raw[i] - mx);
        sum += probs[i];
    }
    if (sum > 0.0) {
        for (auto& p : probs) p = static_cast<float>(p / sum);
    }

    // Top-k by probability.  Pair (prob, idx), partial_sort.
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(n_vocab);
    for (std::size_t i = 0; i < n_vocab; ++i) {
        ranked.emplace_back(probs[i], static_cast<int>(i));
    }
    const int kk = std::min(static_cast<int>(n_vocab),
                            std::max(1, k));
    std::partial_sort(
        ranked.begin(),
        ranked.begin() + kk,
        ranked.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<LogitDist> out;
    out.reserve(static_cast<std::size_t>(kk));
    for (int i = 0; i < kk; ++i) {
        LogitDist d;
        d.token    = "tok_" + std::to_string(ranked[i].second);  // raw token id; UI prefers string but vocab decode would need the model
        d.prob     = ranked[i].first;
        d.delta    = 0.0f;
        d.selected = (i == 0);
        out.push_back(std::move(d));
    }
    return out;
}

// ── getCapabilities ───────────────────────────────────────────────────────────

Model::Capabilities LlamaCppEngine::getCapabilities() const
{
    return Capabilities{
        .has_topology     = true,
        .has_tokenizer    = true,
        .has_state_dict   = false,   // LlamaCppSource for view.tensors still TODO
        .has_attention    = true,
        .has_residual     = true,    // cb_eval captures l_out-{N}
        .has_logit_lens   = true,    // logits captured post-decode
        .has_token_stream = false,   // future — needs llama_decode loop
        .has_captures     = true,
        .has_intervention = false,   // future — needs custom forward
        .has_weight_deltas= false,
        .has_training     = false,
    };
}

// ── view ──────────────────────────────────────────────────────────────────────

const ModelView& LlamaCppEngine::view() const
{
    return m_impl->view;
}

// ── drainEngineLogs ───────────────────────────────────────────────────────────

std::vector<LogEntry> LlamaCppEngine::drainEngineLogs()
{
    std::lock_guard<std::mutex> lk(m_impl->mu);
    return std::exchange(m_impl->pending_logs, {});
}

// ── getProgress ───────────────────────────────────────────────────────────────

Model::Progress LlamaCppEngine::getProgress() const
{
    std::lock_guard<std::mutex> lk(m_impl->prog_mu);
    return m_impl->progress;
}


// ─── Worker thread ────────────────────────────────────────────────────────────

void LlamaCppEngine::Impl::workerRun()
{
    while (true) {
        std::string prompt;

        {
            std::unique_lock<std::mutex> lk(mu);
            wake_cv.wait(lk, [this] {
                return stop_flag.load() || has_pending;
            });
            if (stop_flag.load()) break;
            if (!has_pending) continue;

            prompt      = std::move(pending_prompt);
            has_pending = false;
        }

        if (prompt.empty()) continue;

#ifdef LLM_ENGINE_HAVE_LLAMA_CPP
        llama_model*   lm_snap  = nullptr;
        llama_context* ctx_snap = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu);
            lm_snap  = lm;
            ctx_snap = ctx;
        }
        if (!lm_snap || !ctx_snap) continue;

        const llama_vocab* vocab = llama_model_get_vocab(lm_snap);

        // Tokenize.
        std::vector<llama_token> token_ids(prompt.size() + 32);
        int n = llama_tokenize(vocab,
            prompt.c_str(), static_cast<int32_t>(prompt.size()),
            token_ids.data(), static_cast<int32_t>(token_ids.size()),
            /*add_special=*/true, /*parse_special=*/false);
        if (n < 0) {
            token_ids.resize(static_cast<std::size_t>(-n));
            n = llama_tokenize(vocab,
                prompt.c_str(), static_cast<int32_t>(prompt.size()),
                token_ids.data(), static_cast<int32_t>(token_ids.size()),
                true, false);
        }
        if (n <= 0) {
            pushLog(Severity::Warn, "LlamaCppEngine: tokenize returned 0");
            continue;
        }
        token_ids.resize(static_cast<std::size_t>(n));

        int n_heads = llama_model_n_head(lm_snap);

        LlamaCaptureCtx cap_ctx;
        cap_ctx.n_layers = llama_model_n_layer(lm_snap);
        cap_ctx.n_heads  = n_heads;
        cap_ctx.n_seq    = n;
        cap_ctx.bundle   = std::make_shared<CaptureBundle>();

        // Decode token strings for the bundle.
        cap_ctx.bundle->token_ids.reserve(static_cast<std::size_t>(n));
        cap_ctx.bundle->token_strs.reserve(static_cast<std::size_t>(n));
        for (llama_token tid : token_ids) {
            cap_ctx.bundle->token_ids.push_back(tid);
            char buf[256] = {};
            int nn = llama_token_to_piece(vocab, tid, buf, sizeof(buf) - 1, 0, true);
            cap_ctx.bundle->token_strs.push_back(
                (nn > 0) ? std::string(buf, static_cast<std::size_t>(nn)) : "");
        }

        std::vector<LogEntry> out_logs;
        llama_cpp_run_capture(ctx_snap, lm_snap, token_ids,
                              n_heads, &cap_ctx, out_logs);

        // Atomic-store the completed bundle.
        {
            auto bundle_ptr = std::const_pointer_cast<const CaptureBundle>(
                cap_ctx.bundle);
            view.current.store(bundle_ptr);
        }

        // Push any logs.
        {
            std::lock_guard<std::mutex> lk(mu);
            for (auto& le : out_logs)
                pending_logs.push_back(std::move(le));
        }
#endif  // LLM_ENGINE_HAVE_LLAMA_CPP
    }
}

}  // namespace llmengine
