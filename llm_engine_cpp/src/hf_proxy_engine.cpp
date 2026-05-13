#include "llm_engine/hf_proxy_engine.hpp"
#include "llm_engine/model_view.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
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

    // SamplerWorker thread — drives the heartbeat now, will drive per-frame
    // sample fetches once the WS streaming flow lands (Phase 4).  Lifecycle
    // tied to Impl: spawned in ctor, joined in dtor with stop_flag set so
    // a long sleep returns immediately.
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
    void workerLoop() {
        while (!stop_flag.load()) {
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
            std::unique_lock<std::mutex> lk(mu);
            wake_cv.wait_for(lk, kHeartbeatInterval,
                             [this] { return stop_flag.load(); });
        }
    }
};

namespace {

// Pull an int / float / string out of a JSON object, returning the engine
// sentinel when the key is missing or null.  The backend uses null for
// many optional fields (e.g. rope_theta on models without rotary embeddings)
// — we want those to render as "—" not 0.
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
    // FastAPI endpoints get wired through.
    return Capabilities{
        .has_topology      = true,
        .has_tokenizer     = false,   // Phase 2 wires /tokenize → TokenizerView.encode/decode
        .has_state_dict    = false,
        .has_attention     = false,   // Phase 2
        .has_residual      = false,   // Phase 2
        .has_logit_lens    = false,   // Phase 4 (WS stream)
        .has_token_stream  = false,   // Phase 4
        .has_captures      = false,   // Phase 2 — capture endpoint not wired yet
        .has_intervention  = false,   // Phase 3
        .has_weight_deltas = false,
        .has_training      = false,   // out of scope for HF backend
    };
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

}  // namespace llmengine
