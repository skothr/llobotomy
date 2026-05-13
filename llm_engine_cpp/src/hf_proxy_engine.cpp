#include "llm_engine/hf_proxy_engine.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
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

std::int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
}

}  // namespace

struct HFProxyEngine::Impl {
    std::string           base_url;
    httplib::Client       client;
    std::mutex            mu;                      // guards pending_logs + info
    std::vector<LogEntry> pending_logs;
    ModelInfo             info;                    // populated post-loadCheckpoint

    explicit Impl(std::string url)
        : base_url(std::move(url)), client(base_url) {
        // Connect quickly so a missing backend surfaces a clear error
        // instead of an indefinite hang.  Read/write are generous because
        // the model-load endpoint can take 10s+ for large checkpoints.
        client.set_connection_timeout(5, 0);
        client.set_read_timeout(120, 0);
        client.set_write_timeout(120, 0);
    }

    void log(Severity sev, const char* kind, std::string msg) {
        std::lock_guard<std::mutex> lk(mu);
        pending_logs.push_back({now_ms(), sev, kind, std::move(msg)});
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
                m_impl->info = std::move(parsed);
                m_impl->log(Severity::Info, "hf",
                            "topology: " + std::to_string(m_impl->info.nLayers) +
                            " layers · " + std::to_string(m_impl->info.nHeads) +
                            " heads · d_model=" +
                            std::to_string(m_impl->info.dModel));
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
        m_impl->info = {};
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
    return m_impl->info;
}

}  // namespace llmengine
