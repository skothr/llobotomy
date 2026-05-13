#include "llm_engine/hf_proxy_engine.hpp"
#include "llm_engine/model_view.hpp"
#include "llm_engine/capture.hpp"
#include "llm_engine/tensor_source.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using nlohmann::json;

namespace llmengine {

namespace {

// One fixed session name on the backend side.  The C++ Model interface is
// single-active-checkpoint; the backend supports multiple sessions but we
// only use one slot, named consistently so an orphaned session from a
// crashed prior process can be re-claimed by DELETE before re-POST.
constexpr const char* kSessionName = "llobotomy";

// Default quantisation mode passed to the backend's LoadRequest.  Matches
// the React frontend's default; backend will fall back / warn if a model
// can't quantise to nf4.  Override via LLOB_BACKEND_MODE env var at the
// MakeBackend factory site if needed (Phase 2).
constexpr const char* kDefaultLoadMode = "nf4";

// SamplerWorker cadence.  The worker thread sleeps `kHeartbeatInterval`
// between liveness pings to /api/sessions; on a successful response the
// engine's "live" flag stays true, on failure it goes false and the UI
// renders the device tag with a "(down)" suffix.  10s is a balance —
// fast enough that user reconnects feel responsive, slow enough not to
// spam the backend.
constexpr auto kHeartbeatInterval = std::chrono::seconds(10);

std::int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
}

// JSON helpers — kept here (before Impl) so doCapture inside Impl can
// call them.  The second anonymous namespace block below was their
// original home; they've been hoisted so the definition precedes use.
int json_int(const json& j, const char* key, int fallback = kNoInt) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return fallback;
    return it->get<int>();
}
std::int64_t json_i64(const json& j, const char* key, std::int64_t fallback = kNoSize) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return fallback;
    return it->get<std::int64_t>();
}
float json_float(const json& j, const char* key, float fallback = kNoFloat) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return fallback;
    return it->get<float>();
}
std::string json_string(const json& j, const char* key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return {};
    return it->get<std::string>();
}

}  // namespace

struct HFProxyEngine::Impl {
    std::string           base_url;
    // Two HTTP clients so the worker's heartbeat doesn't serialize behind
    // a slow loadCheckpoint POST on the UI thread.  cpp-httplib's
    // httplib::Client is not internally thread-safe (each instance owns
    // a single keep-alive connection); separate clients per logical
    // caller is the recommended pattern.
    httplib::Client       client;
    httplib::Client       worker_client;
    std::mutex            mu;                      // guards everything below
    std::vector<LogEntry> pending_logs;
    // Unified data structure.  loadCheckpoint populates view.provenance +
    // view.topology; unloadCheckpoint resets them.  Future phases write
    // captures + surgery into the same struct.  view() returns a const
    // reference; concurrent backend writes happen under `mu`, concurrent
    // UI reads accept eventual consistency for the scalar fields.
    ModelView             view;
    bool                  backend_live = false;    // last heartbeat result
    std::int64_t          last_contact_ms = 0;     // 0 ⇒ never reached

    // ── Phase 2 capture state ────────────────────────────────────────────
    // setActivePrompt stores the prompt here then wakes the worker.  The
    // worker reads it (under mu), runs the HTTP fetch, and stores the
    // resulting CaptureBundle into view.current via atomic swap.
    std::string   pending_prompt;        // "" ⇒ nothing pending
    std::string   active_hash;           // hash of the currently-loaded capture

    // SamplerWorker thread — drives heartbeat (Phase 1) and capture
    // jobs (Phase 2).  Lifecycle tied to Impl.
    std::atomic<bool>           stop_flag{false};
    std::condition_variable     wake_cv;
    std::thread                 worker;

    explicit Impl(std::string url)
        : base_url(std::move(url)),
          client(base_url),
          worker_client(base_url) {
        // Connect quickly so a missing backend surfaces a clear error
        // instead of an indefinite hang.  Read/write are generous because
        // the model-load endpoint can take 10s+ for large checkpoints.
        client.set_connection_timeout(5, 0);
        client.set_read_timeout(120, 0);
        client.set_write_timeout(120, 0);
        // Worker uses tighter timeouts since heartbeat queries are quick.
        worker_client.set_connection_timeout(2, 0);
        worker_client.set_read_timeout(5, 0);
        worker_client.set_write_timeout(5, 0);

        worker = std::thread([this] { workerLoop(); });
    }

    ~Impl() {
        stop_flag.store(true);
        wake_cv.notify_all();
        if (worker.joinable()) worker.join();
    }

    void log(Severity sev, const char* kind, std::string msg) {
        std::lock_guard<std::mutex> lk(mu);
        pending_logs.push_back({now_ms(), sev, kind, std::move(msg)});
    }

    // Heartbeat: GET /api/sessions every kHeartbeatInterval; on transition
    // (down→up or up→down), emit a single log line so the UI's status
    // toast reflects the change.  The actual response payload is ignored —
    // a 200 means the backend is up enough to talk.
    //
    // Phase 2: between heartbeat sleeps, drain pending_prompt.  The worker
    // POSTs /capture, then builds a minimal CaptureBundle from the response
    // (token strings + shape) and swaps it into view.current.  Subsequent
    // UI getter calls lazily fill in attention / residual / qkv.
    void workerLoop() {
        while (!stop_flag.load()) {
            // ── Heartbeat ────────────────────────────────────────────────
            const auto res = worker_client.Get("/api/sessions");
            const bool live = (res && res->status == 200);
            const std::int64_t t = now_ms();
            bool changed = false;
            {
                std::lock_guard<std::mutex> lk(mu);
                if (live != backend_live) {
                    backend_live = live;
                    changed = true;
                }
                if (live) last_contact_ms = t;
            }
            if (changed) {
                log(live ? Severity::Info : Severity::Warn,
                    "hf",
                    live ? "backend reachable"
                         : "backend unreachable");
            }

            // ── Capture job ──────────────────────────────────────────────
            std::string prompt_to_capture;
            {
                std::lock_guard<std::mutex> lk(mu);
                prompt_to_capture = pending_prompt;
                pending_prompt.clear();
            }
            if (!prompt_to_capture.empty()) {
                doCapture(prompt_to_capture);
            }

            std::unique_lock<std::mutex> lk(mu);
            wake_cv.wait_for(lk, kHeartbeatInterval,
                             [this] { return stop_flag.load() || !pending_prompt.empty(); });
        }
    }

    // POST /api/sessions/{n}/capture → GET /api/sessions/{n}/capture/{h}/tokens
    // Builds a CaptureBundle from the shape metadata + token list, then
    // atomically installs it into view.current.
    void doCapture(const std::string& prompt) {
        const std::string capture_path =
            std::string("/api/sessions/") + kSessionName + "/capture";
        const json body = {{"prompt", prompt}};

        auto cap_res = worker_client.Post(
            capture_path.c_str(), body.dump(), "application/json");

        if (!cap_res) {
            log(Severity::Warn, "hf",
                std::string("capture POST failed: ") +
                httplib::to_string(cap_res.error()));
            return;
        }
        if (cap_res->status >= 400) {
            log(Severity::Warn, "hf",
                "capture POST returned " + std::to_string(cap_res->status) +
                ": " + cap_res->body);
            return;
        }

        json j;
        try { j = json::parse(cap_res->body); }
        catch (const std::exception& e) {
            log(Severity::Warn, "hf",
                std::string("capture POST parse failed: ") + e.what());
            return;
        }

        const std::string hash = json_string(j, "prompt_hash");
        const int layers   = json_int(j, "layers");
        const int heads    = json_int(j, "heads");
        const int seq_len  = json_int(j, "seq_len");
        if (hash.empty() || layers <= 0 || heads <= 0 || seq_len <= 0) {
            log(Severity::Warn, "hf", "capture POST: unexpected shape " + cap_res->body);
            return;
        }

        // Fetch token strings.
        const std::string tokens_path =
            std::string("/api/sessions/") + kSessionName +
            "/capture/" + hash + "/tokens";
        auto tok_res = worker_client.Get(tokens_path.c_str());

        std::vector<std::string> token_strs;
        if (tok_res && tok_res->status == 200) {
            try {
                json tj = json::parse(tok_res->body);
                for (const auto& s : tj.at("tokens")) {
                    token_strs.push_back(s.get<std::string>());
                }
            } catch (...) {
                log(Severity::Warn, "hf", "capture /tokens parse failed");
            }
        }

        // Build a minimal CaptureBundle.  Attention / residual / qkv
        // handles are filled lazily by the getter methods on UI calls.
        auto bundle = std::make_shared<CaptureBundle>();
        bundle->prompt_hash = hash;
        bundle->token_strs  = std::move(token_strs);

        // Install into view.current (lock-free for the UI, mutex for us
        // because we also update active_hash).
        {
            std::lock_guard<std::mutex> lk(mu);
            active_hash = hash;
            view.captures[hash] = bundle;
        }
        view.current.store(bundle);

        log(Severity::Info, "hf",
            "capture ready: hash=" + hash +
            " layers=" + std::to_string(layers) +
            " heads=" + std::to_string(heads) +
            " seq=" + std::to_string(seq_len));
    }
};

namespace {

// Translate a SessionInfoResponse JSON into engine-side ModelInfo.  Field
// names are normalised: num_layers → nLayers, hidden_size → dModel, etc.
// dHead is derived (hidden_size / num_heads) since the backend doesn't
// emit it directly.
ModelInfo parseSessionInfo(const json& j, const std::string& display_name) {
    ModelInfo m;
    m.name         = display_name;
    m.nLayers      = json_int   (j, "num_layers");
    m.nHeads       = json_int   (j, "num_heads");
    m.nKvHeads     = json_int   (j, "num_kv_heads", m.nHeads);
    m.dModel       = json_int   (j, "hidden_size");
    m.dMlp         = json_int   (j, "intermediate_size");
    m.vocab        = json_int   (j, "vocab_size");
    m.maxPos       = json_int   (j, "max_position_embeddings");
    m.ropeTheta    = json_float (j, "rope_theta");
    m.totalParams  = json_i64   (j, "total_params");
    m.chatTemplate = json_string(j, "chat_template");
    m.bosToken     = json_string(j, "bos_token");
    m.eosToken     = json_string(j, "eos_token");
    if (m.nHeads > 0 && m.dModel > 0) m.dHead = m.dModel / m.nHeads;
    return m;
}

}  // namespace

HFProxyEngine::HFProxyEngine(std::string base_url)
    : m_impl(std::make_unique<Impl>(std::move(base_url))) {
    m_impl->log(Severity::Info, "hf",
                "HFProxyEngine ready (base=" + m_impl->base_url + ")");
}

HFProxyEngine::~HFProxyEngine() = default;

Model::CheckpointResult HFProxyEngine::loadCheckpoint(std::string_view path) {
    const std::string model_id(path);
    if (model_id.empty()) {
        return {false, "loadCheckpoint: empty model_id"};
    }

    // Defensively drop any session left over from a previous run.  404 is
    // expected (and fine) when no orphan exists.
    {
        const std::string del_path = std::string("/api/sessions/") + kSessionName;
        auto res = m_impl->client.Delete(del_path.c_str());
        if (res && res->status == 200) {
            m_impl->log(Severity::Debug, "hf",
                        "removed stale session before reload");
        }
    }

    const json body = {
        {"name",     kSessionName},
        {"model_id", model_id},
        {"mode",     kDefaultLoadMode},
    };
    auto res = m_impl->client.Post(
        "/api/sessions",
        body.dump(),
        "application/json");

    if (!res) {
        std::string err = "POST /api/sessions failed (";
        err += httplib::to_string(res.error());
        err += ") base=" + m_impl->base_url;
        m_impl->log(Severity::Error, "hf", err);
        return {false, err};
    }
    if (res->status >= 400) {
        std::string err = "POST /api/sessions returned " +
                          std::to_string(res->status) + ": " + res->body;
        m_impl->log(Severity::Error, "hf", err);
        return {false, err};
    }

    m_impl->log(Severity::Info, "hf",
                "loaded session '" + std::string(kSessionName) +
                "' model=" + model_id);

    // Cache the full session info so getModelInfo() returns real layer
    // counts / hidden size etc. without an extra round-trip per query.
    {
        const std::string info_path =
            std::string("/api/sessions/") + kSessionName + "/info";
        auto info_res = m_impl->client.Get(info_path.c_str());
        if (info_res && info_res->status == 200) {
            try {
                ModelInfo parsed = parseSessionInfo(
                    json::parse(info_res->body), model_id);
                std::lock_guard<std::mutex> lk(m_impl->mu);
                m_impl->view.topology   = std::move(parsed);
                m_impl->view.provenance = {
                    .path         = model_id,
                    .format       = "hf-proxy",
                    .content_hash = "",
                    .source_label = "HuggingFace via FastAPI (" + m_impl->base_url + ")",
                };
                // Capability mirror for the path API.  Phase 2: topology,
                // tokenizer (via /capture + /tokens), attention matrices,
                // residual summaries, QKV stats, and captures are wired.
                m_impl->view.capabilities = Capabilities{
                    .has_topology      = true,
                    .has_tokenizer     = true,   // token strings from /capture/tokens
                    .has_state_dict    = false,
                    .has_attention     = true,
                    .has_residual      = true,
                    .has_logit_lens    = false,
                    .has_token_stream  = false,
                    .has_captures      = true,
                    .has_intervention  = false,
                    .has_weight_deltas = false,
                    .has_training      = false,
                };
                m_impl->log(Severity::Info, "hf",
                            "topology: " + std::to_string(m_impl->view.topology.nLayers) +
                            " layers · " + std::to_string(m_impl->view.topology.nHeads) +
                            " heads · d_model=" +
                            std::to_string(m_impl->view.topology.dModel));
            } catch (const std::exception& e) {
                m_impl->log(Severity::Warn, "hf",
                            std::string("session/info parse failed: ") + e.what());
            }
        } else {
            const std::string status =
                info_res ? std::to_string(info_res->status) : "no-response";
            m_impl->log(Severity::Warn, "hf",
                        "session/info GET failed (status=" + status + ")");
        }
    }
    return {true, ""};
}

void HFProxyEngine::unloadCheckpoint() {
    {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        // ModelView::clear() resets every field — topology, provenance,
        // tensors, captures, surgery, derived.  Required: surgery deltas
        // tied to model-A tensors must not survive into a model-B session.
        m_impl->view.clear();
    }
    const std::string del_path = std::string("/api/sessions/") + kSessionName;
    auto res = m_impl->client.Delete(del_path.c_str());
    if (!res) {
        m_impl->log(Severity::Warn, "hf",
                    "unloadCheckpoint: DELETE failed (" +
                    std::string(httplib::to_string(res.error())) + ")");
        return;
    }
    if (res->status == 404) {
        m_impl->log(Severity::Debug, "hf", "unloadCheckpoint: no active session");
    } else if (res->status >= 400) {
        m_impl->log(Severity::Warn, "hf",
                    "unloadCheckpoint: status=" + std::to_string(res->status));
    } else {
        m_impl->log(Severity::Info, "hf", "unloaded session");
    }
}

std::vector<LogEntry> HFProxyEngine::drainEngineLogs() {
    std::lock_guard<std::mutex> lk(m_impl->mu);
    std::vector<LogEntry> out;
    out.swap(m_impl->pending_logs);
    return out;
}

ModelInfo HFProxyEngine::getModelInfo() {
    std::lock_guard<std::mutex> lk(m_impl->mu);
    return m_impl->view.topology;
}

const ModelView& HFProxyEngine::view() const {
    // No lock here — ModelView reads happen at frame rate.  Topology and
    // provenance are scalar fields whose torn reads still produce sane
    // values (the most you'll see during a swap is one stale scalar for
    // a single frame).  Anything that needs strict atomicity (the
    // capture pointer in `view.current`) lives behind its own atomic.
    return m_impl->view;
}

Model::Capabilities HFProxyEngine::getCapabilities() const {
    // Phase 1: we own the topology + log fan-in only.  Subsequent phases
    // light up attention / residual / captures / intervention as their
    // FastAPI endpoints get wired through.  Mirror is also written into
    // view.capabilities at loadCheckpoint so path-API consumers can
    // resolve `capabilities/has_X` without going through the virtual.
    std::lock_guard<std::mutex> lk(m_impl->mu);
    return m_impl->view.capabilities;
}

EngineMetrics HFProxyEngine::getEngineMetrics() {
    EngineMetrics m;
    std::lock_guard<std::mutex> lk(m_impl->mu);
    if (m_impl->backend_live) {
        if (m_impl->view.topology.nLayers > 0) {
            // Loaded checkpoint — render as "hf · <model_id>".
            m.device = "hf · " + m_impl->view.topology.name;
        } else {
            m.device = "hf · idle";
        }
        m.dtype  = "fp16";   // backend default; refine when load mode is parsed
    } else {
        const std::int64_t since = m_impl->last_contact_ms == 0
            ? -1
            : (now_ms() - m_impl->last_contact_ms);
        if (since < 0) m.device = "hf · unreachable";
        else           m.device = "hf · down " + std::to_string(since / 1000) + "s";
        m.dtype = "—";
    }
    return m;
}

// ── Phase 2 — capture-driven accessors ────────────────────────────────────

void HFProxyEngine::setActivePrompt(std::string_view prompt) {
    // Copy (caller's borrow ends after this call) then wake the worker.
    std::lock_guard<std::mutex> lk(m_impl->mu);
    m_impl->pending_prompt = std::string(prompt);
    m_impl->wake_cv.notify_all();
}

std::vector<std::string> HFProxyEngine::getCurrentTokens() {
    auto bundle = m_impl->view.current.load();
    if (!bundle) return {};
    return bundle->token_strs;
}

std::vector<std::vector<float>> HFProxyEngine::getAttentionPattern(
    int layer, int head, [[maybe_unused]] int seqLen,
    [[maybe_unused]] HeadBias bias)
{
    auto bundle = m_impl->view.current.load();
    if (!bundle) return {};

    const auto key = std::make_pair(layer, head);

    // Cache-hit: handle already in the bundle.
    {
        // CaptureBundle's maps are written only from the worker thread or from
        // this UI-thread getter (first call).  We use a per-Impl mutex to guard
        // concurrent getters on the same bundle.
        std::lock_guard<std::mutex> lk(m_impl->mu);
        auto it = bundle->attention.find(key);
        if (it != bundle->attention.end() && it->second.readable()) {
            // Read from the cached handle.
            const auto& h = it->second;
            std::size_t rows = static_cast<std::size_t>(h.shape[0]);
            std::size_t cols = static_cast<std::size_t>(h.shape[1]);
            return h.read_slice_2d(0, rows, 0, cols);
        }
    }

    // Cache miss: fetch from backend synchronously (first call per cell).
    std::string hash;
    { std::lock_guard<std::mutex> lk(m_impl->mu); hash = m_impl->active_hash; }
    if (hash.empty()) return {};

    const std::string path =
        std::string("/api/sessions/") + kSessionName +
        "/capture/" + hash +
        "/attention?layer=" + std::to_string(layer) +
        "&head=" + std::to_string(head);

    auto res = m_impl->client.Get(path.c_str());
    if (!res || res->status != 200) {
        m_impl->log(Severity::Warn, "hf",
                    "getAttentionPattern: fetch failed layer=" + std::to_string(layer) +
                    " head=" + std::to_string(head));
        return {};
    }

    std::vector<std::vector<float>> matrix;
    try {
        const json j = json::parse(res->body);
        for (const auto& row : j.at("matrix")) {
            std::vector<float> r;
            r.reserve(row.size());
            for (const auto& v : row) r.push_back(v.get<float>());
            matrix.push_back(std::move(r));
        }
    } catch (const std::exception& e) {
        m_impl->log(Severity::Warn, "hf",
                    std::string("getAttentionPattern parse failed: ") + e.what());
        return {};
    }

    if (matrix.empty()) return {};

    // Cache into the bundle so the next call is free.
    const std::int64_t rows = static_cast<std::int64_t>(matrix.size());
    const std::int64_t cols = static_cast<std::int64_t>(matrix[0].size());
    std::vector<float> flat;
    flat.reserve(static_cast<std::size_t>(rows * cols));
    for (const auto& row : matrix)
        for (float v : row)
            flat.push_back(v);

    auto src = InMemoryTensorSource::from_floats(std::move(flat));
    TensorHandle h;
    h.source      = src;
    h.name        = "attn." + std::to_string(layer) + "." + std::to_string(head);
    h.dtype       = DType::F32;
    h.shape       = {rows, cols};
    h.byte_offset = 0;
    h.byte_length = src->size_bytes();
    h.contiguous  = true;

    {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        // Cast away const — we own the bundle exclusively via the captures map.
        const_cast<CaptureBundle*>(bundle.get())->attention[key] = std::move(h);
    }

    return matrix;
}

ResidualSummary HFProxyEngine::getResidualSummary(int layer) {
    auto bundle = m_impl->view.current.load();
    if (!bundle) return {};

    // Check if we have a pre-computed summary.
    {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        auto it = bundle->residual_summary.find(layer);
        if (it != bundle->residual_summary.end() && it->second) {
            return *it->second;
        }
    }

    std::string hash;
    { std::lock_guard<std::mutex> lk(m_impl->mu); hash = m_impl->active_hash; }
    if (hash.empty()) return {};

    const std::string path =
        std::string("/api/sessions/") + kSessionName +
        "/capture/" + hash +
        "/residual?layer=" + std::to_string(layer);

    auto res = m_impl->client.Get(path.c_str());
    if (!res || res->status != 200) {
        m_impl->log(Severity::Warn, "hf",
                    "getResidualSummary: fetch failed layer=" + std::to_string(layer));
        return {};
    }

    ResidualSummary s;
    try {
        const json j = json::parse(res->body);
        s.attn_out_norm = json_float(j, "attn_out_norm");
        s.mlp_out_norm  = json_float(j, "mlp_out_norm");
        s.resid_norm    = json_float(j, "resid_norm");
        s.cos_prev      = json_float(j, "cos_prev");
        s.kurtosis      = json_float(j, "kurtosis");
        s.rank_eff      = json_int  (j, "rank_eff");
        s.rank_full     = json_int  (j, "rank_full");
    } catch (const std::exception& e) {
        m_impl->log(Severity::Warn, "hf",
                    std::string("getResidualSummary parse failed: ") + e.what());
        return {};
    }

    // Cache.
    {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        const_cast<CaptureBundle*>(bundle.get())->residual_summary[layer] =
            std::make_shared<ResidualSummary>(s);
    }
    return s;
}

QKVStats HFProxyEngine::getQKVStats(int layer, int head, int token) {
    auto bundle = m_impl->view.current.load();
    if (!bundle) return {};

    std::string hash;
    { std::lock_guard<std::mutex> lk(m_impl->mu); hash = m_impl->active_hash; }
    if (hash.empty()) return {};

    const std::string path =
        std::string("/api/sessions/") + kSessionName +
        "/capture/" + hash +
        "/qkv?layer=" + std::to_string(layer) +
        "&head=" + std::to_string(head) +
        "&token=" + std::to_string(token);

    auto res = m_impl->client.Get(path.c_str());
    if (!res || res->status != 200) {
        m_impl->log(Severity::Warn, "hf",
                    "getQKVStats: fetch failed L=" + std::to_string(layer) +
                    " H=" + std::to_string(head) +
                    " T=" + std::to_string(token));
        return {};
    }

    QKVStats q;
    try {
        const json j  = json::parse(res->body);
        q.q_norm       = json_float(j, "q_norm");
        q.k_norm       = json_float(j, "k_norm");
        q.v_norm       = json_float(j, "v_norm");
        q.attn_to_bos  = json_float(j, "attn_to_bos");
        q.attn_to_self = json_float(j, "attn_to_self");
        q.attn_to_prev = json_float(j, "attn_to_prev");
    } catch (const std::exception& e) {
        m_impl->log(Severity::Warn, "hf",
                    std::string("getQKVStats parse failed: ") + e.what());
        return {};
    }
    return q;
}

}  // namespace llmengine
