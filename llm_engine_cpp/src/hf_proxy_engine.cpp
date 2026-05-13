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
#include <algorithm>

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

    // ── Phase 3 intervention state ───────────────────────────────────────
    // pending_intervention is written by setAblation/setSteering/
    // clearSteering (under mu).  dirty_intervention signals the worker
    // to reconcile pending vs. installed (m_view.surgery).
    //
    // Steering is handled separately: dirty_steering_op selects the
    // pending operation (Install vs. Clear).
    enum class SteeringOp { None, Install, Clear };

    struct PendingIntervention {
        std::vector<AttentionHeadRef> heads;
        std::vector<ComponentRef>     components;
    };

    PendingIntervention  pending_intervention;
    bool                 dirty_intervention = false;  // ablation set changed
    SteeringOp           dirty_steering_op  = SteeringOp::None;
    SteeringConfig       pending_steering;            // valid when dirty_steering_op == Install

    // ── Phase 4 — WS stream state ────────────────────────────────────────
    // ws_base is the base_url with "http" replaced by "ws" (or "https"→"wss").
    // Used by getLogitLensTrajectory / getOutputLogits to build full WS URLs.
    // Computed once at construction.
    std::string ws_base;

    // SamplerWorker thread — drives heartbeat (Phase 1) and capture
    // jobs (Phase 2).  Lifecycle tied to Impl.
    std::atomic<bool>           stop_flag{false};
    std::condition_variable     wake_cv;
    std::thread                 worker;

    explicit Impl(std::string url)
        : base_url(std::move(url)),
          client(base_url),
          worker_client(base_url) {
        // Derive WS base URL from the HTTP base URL (http→ws, https→wss).
        ws_base = base_url;
        if (ws_base.substr(0, 8) == "https://") {
            ws_base = "wss://" + ws_base.substr(8);
        } else if (ws_base.substr(0, 7) == "http://") {
            ws_base = "ws://" + ws_base.substr(7);
        }
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

            // ── Intervention job ─────────────────────────────────────────
            bool do_ablation = false;
            PendingIntervention ablation_snap;
            SteeringOp steer_op = SteeringOp::None;
            SteeringConfig steer_snap;
            {
                std::lock_guard<std::mutex> lk(mu);
                if (dirty_intervention) {
                    do_ablation       = true;
                    ablation_snap     = pending_intervention;
                    dirty_intervention = false;
                }
                if (dirty_steering_op != SteeringOp::None) {
                    steer_op       = dirty_steering_op;
                    steer_snap     = pending_steering;
                    dirty_steering_op = SteeringOp::None;
                }
            }
            if (do_ablation) {
                doAblation(ablation_snap);
            }
            if (steer_op == SteeringOp::Install) {
                doInstallSteering(steer_snap);
            } else if (steer_op == SteeringOp::Clear) {
                doClearSteering();
            }

            std::unique_lock<std::mutex> lk(mu);
            wake_cv.wait_for(lk, kHeartbeatInterval,
                             [this] {
                                 return stop_flag.load()
                                     || !pending_prompt.empty()
                                     || dirty_intervention
                                     || dirty_steering_op != SteeringOp::None;
                             });
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

    // ── Phase 3 — intervention helpers (run on the worker thread) ────────

    // POST /api/sessions/{n}/surgery for a single op, return true on success.
    bool postSurgeryOp(const std::string& operation, const json& params) {
        const std::string path =
            std::string("/api/sessions/") + kSessionName + "/surgery";
        const json body = {{"operation", operation}, {"params", params}};
        auto res = worker_client.Post(path.c_str(), body.dump(), "application/json");
        if (!res) {
            log(Severity::Warn, "hf",
                "surgery POST failed (op=" + operation + "): " +
                std::string(httplib::to_string(res.error())));
            return false;
        }
        if (res->status >= 400) {
            log(Severity::Warn, "hf",
                "surgery POST " + operation + " returned " +
                std::to_string(res->status) + ": " + res->body);
            return false;
        }
        return true;
    }

    // POST /surgery/commit — flush all staged ops to the model weights.
    bool postSurgeryCommit() {
        const std::string path =
            std::string("/api/sessions/") + kSessionName + "/surgery/commit";
        auto res = worker_client.Post(path.c_str(), "{}", "application/json");
        if (!res) {
            log(Severity::Warn, "hf",
                "surgery/commit POST failed: " +
                std::string(httplib::to_string(res.error())));
            return false;
        }
        if (res->status >= 400) {
            log(Severity::Warn, "hf",
                "surgery/commit returned " + std::to_string(res->status) +
                ": " + res->body);
            return false;
        }
        return true;
    }

    // Diff pending ablation set against currently-installed, issue POSTs,
    // commit on success, then update m_view.surgery.
    void doAblation(const PendingIntervention& snap) {
        // Collect what's currently installed (snapshot under lock).
        std::vector<AttentionHeadRef> installed_heads;
        std::vector<ComponentRef>     installed_comps;
        {
            std::lock_guard<std::mutex> lk(mu);
            installed_heads = view.surgery.ablated_heads;
            installed_comps = view.surgery.ablated_components;
        }

        // Compute added heads: in snap.heads but not in installed.
        std::vector<AttentionHeadRef> added_heads;
        for (const auto& h : snap.heads) {
            bool found = false;
            for (const auto& ih : installed_heads) {
                if (ih == h) { found = true; break; }
            }
            if (!found) added_heads.push_back(h);
        }
        // Compute removed heads: in installed but not in snap.
        std::vector<AttentionHeadRef> removed_heads;
        for (const auto& ih : installed_heads) {
            bool found = false;
            for (const auto& h : snap.heads) {
                if (h == ih) { found = true; break; }
            }
            if (!found) removed_heads.push_back(ih);
        }

        // For removed heads we'd need to POST surgery/revert by index — which
        // requires knowing the index in the backend's pending queue. Since the
        // C++ side doesn't maintain that index, and the Python backend doesn't
        // yet support reverting individual ops atomically, use the simpler
        // strategy: clear all pending on the backend (GET pending, DELETE each
        // one), then re-stage the full snap set.  This is safe because a
        // /surgery/commit has not been issued for the pending queue yet —
        // we're still in "staged" state.
        //
        // If there's nothing to remove (purely additive change) we skip the
        // clear and just stage the additions — cheaper.
        const bool needs_clear = !removed_heads.empty()
                                 || snap.components != installed_comps;

        if (needs_clear) {
            // Clear the backend's pending surgery queue.
            const std::string pending_path =
                std::string("/api/sessions/") + kSessionName + "/surgery/pending";
            auto pending_res = worker_client.Get(pending_path.c_str());
            if (!pending_res || pending_res->status != 200) {
                log(Severity::Warn, "hf",
                    "ablation: could not fetch pending ops — aborting");
                return;
            }
            int count = 0;
            try {
                count = static_cast<int>(
                    json::parse(pending_res->body).at("pending").size());
            } catch (...) {}

            // DELETE /surgery/last N times to drain the pending queue.
            for (int i = 0; i < count; ++i) {
                const std::string last_path =
                    std::string("/api/sessions/") + kSessionName +
                    "/surgery/last";
                auto del_res = worker_client.Delete(last_path.c_str());
                if (!del_res || del_res->status >= 400) {
                    log(Severity::Warn, "hf",
                        "ablation: failed to delete pending op during clear");
                    // Continue anyway — queue may already be partially cleared.
                }
            }
            // After clearing: stage the full snap set.
            added_heads     = snap.heads;
        }

        // Stage newly-added attention heads.
        for (const auto& h : added_heads) {
            const json params = {{"layer", h.layer}, {"heads", json::array({h.head})}};
            if (!postSurgeryOp("zero_heads", params)) {
                log(Severity::Warn, "hf",
                    "ablation: failed to stage zero_heads L=" +
                    std::to_string(h.layer) + " H=" + std::to_string(h.head));
                return;
            }
        }

        // Stage newly-added components (zero_mlp or zero_attention).
        for (const auto& c : snap.components) {
            const bool is_mlp  = (c.component == "mlp");
            const bool is_attn = (c.component == "attn");
            if (!is_mlp && !is_attn) {
                log(Severity::Warn, "hf",
                    "ablation: component '" + c.component +
                    "' not stageable via surgery — skipped");
                continue;
            }
            const std::string op = is_mlp ? "zero_mlp" : "zero_attention";
            const json params    = {{"layer", c.layer}};
            if (!postSurgeryOp(op, params)) {
                log(Severity::Warn, "hf",
                    "ablation: failed to stage " + op +
                    " L=" + std::to_string(c.layer));
                return;
            }
        }

        // No ops staged (the snap is empty and nothing was queued).
        // Commit regardless — the Python side returns 409 if nothing is
        // pending, which we treat as a non-fatal warn.
        if (snap.heads.empty() && snap.components.empty() && !needs_clear) {
            // Nothing to do.
            std::lock_guard<std::mutex> lk(mu);
            view.surgery.ablated_heads.clear();
            view.surgery.ablated_components.clear();
            return;
        }

        if (!postSurgeryCommit()) {
            log(Severity::Warn, "hf",
                "ablation: commit failed — m_view.surgery unchanged (stale)");
            return;
        }

        // Commit succeeded: update m_view.surgery and invalidate the capture.
        {
            std::lock_guard<std::mutex> lk(mu);
            view.surgery.ablated_heads      = snap.heads;
            view.surgery.ablated_components = snap.components;
        }
        view.current.store(nullptr);  // capture invalidated — UI re-runs prompt

        log(Severity::Info, "hf",
            "ablation committed: " +
            std::to_string(snap.heads.size()) + " head(s), " +
            std::to_string(snap.components.size()) + " component(s)");
    }

    // POST /api/sessions/{n}/steering
    void doInstallSteering(const SteeringConfig& cfg) {
        const std::string path =
            std::string("/api/sessions/") + kSessionName + "/steering";
        const json body = {
            {"layer",  cfg.layer},
            {"alpha",  cfg.alpha},
            {"source", cfg.source},
        };
        auto res = worker_client.Post(path.c_str(), body.dump(), "application/json");
        if (!res) {
            log(Severity::Warn, "hf",
                "steering POST failed: " +
                std::string(httplib::to_string(res.error())));
            return;
        }
        if (res->status >= 400) {
            log(Severity::Warn, "hf",
                "steering POST returned " + std::to_string(res->status) +
                ": " + res->body);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            view.surgery.steering = cfg;
        }
        view.current.store(nullptr);
        log(Severity::Info, "hf",
            "steering installed: layer=" + cfg.layer +
            " alpha=" + std::to_string(cfg.alpha));
    }

    // DELETE /api/sessions/{n}/steering
    void doClearSteering() {
        const std::string path =
            std::string("/api/sessions/") + kSessionName + "/steering";
        auto res = worker_client.Delete(path.c_str());
        if (!res) {
            log(Severity::Warn, "hf",
                "steering DELETE failed: " +
                std::string(httplib::to_string(res.error())));
            return;
        }
        if (res->status >= 400 && res->status != 404) {
            log(Severity::Warn, "hf",
                "steering DELETE returned " + std::to_string(res->status) +
                ": " + res->body);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            view.surgery.steering = SteeringConfig{};
        }
        view.current.store(nullptr);
        log(Severity::Info, "hf", "steering cleared");
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
                // Capability mirror for the path API.  Phase 3: intervention
                // endpoints are wired (setAblation / setSteering /
                // clearSteering via /surgery + /steering).
                // Phase 4: logit-lens + token-stream via WebSocket;
                // capabilities are flipped to true here to advertise
                // the hooks.  The first actual WS call may still fail
                // gracefully (unreachable backend) — the UI will then
                // show empty panels rather than crashing.
                m_impl->view.capabilities = Capabilities{
                    .has_topology      = true,
                    .has_tokenizer     = true,   // token strings from /capture/tokens
                    .has_state_dict    = false,
                    .has_attention     = true,
                    .has_residual      = true,
                    .has_logit_lens    = true,   // Phase 4: WS logit-lens
                    .has_token_stream  = true,   // Phase 4: WS generate
                    .has_captures      = true,
                    .has_intervention  = true,   // Phase 3: setAblation / setSteering
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

// ── Phase 3 — intervention mutators ───────────────────────────────────────

void HFProxyEngine::setAblation(
    const std::vector<std::string>& head_canonical,
    const std::vector<std::string>& component_canonical)
{
    Impl::PendingIntervention pi;

    for (const auto& name : head_canonical) {
        auto ref = AttentionHeadRef::parse(name);
        if (!ref) {
            m_impl->log(Severity::Warn, "hf",
                        "setAblation: failed to parse head ref '" + name + "' — skipped");
            continue;
        }
        pi.heads.push_back(*ref);
    }
    for (const auto& name : component_canonical) {
        auto ref = ComponentRef::parse(name);
        if (!ref) {
            m_impl->log(Severity::Warn, "hf",
                        "setAblation: failed to parse component ref '" + name + "' — skipped");
            continue;
        }
        pi.components.push_back(*ref);
    }

    {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        m_impl->pending_intervention = std::move(pi);
        m_impl->dirty_intervention   = true;
    }
    m_impl->wake_cv.notify_all();
}

void HFProxyEngine::setSteering(const SteeringConfig& cfg) {
    {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        m_impl->pending_steering   = cfg;
        m_impl->dirty_steering_op  = Impl::SteeringOp::Install;
    }
    m_impl->wake_cv.notify_all();
}

void HFProxyEngine::clearSteering() {
    {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        m_impl->dirty_steering_op = Impl::SteeringOp::Clear;
    }
    m_impl->wake_cv.notify_all();
}

// ── Phase 4 — WebSocket stream implementations ────────────────────────────

// getLogitLensTrajectory: open WS /ws/sessions/{n}/logit-lens, send
// {prompt, top_k: kLayers}, accumulate per-layer frames into LogitLensRow
// vector.  Returns {} on WS connection failure or empty capture.
//
// The `token` parameter selects which token position to inspect.  The
// backend's logit-lens streams predictions for all positions simultaneously
// (as arrays of per-position top-k); we extract the [token] position from
// each layer frame.
std::vector<LogitLensRow> HFProxyEngine::getLogitLensTrajectory(
    int token, int kLayers)
{
    // We need a captured prompt to send.  Use the current bundle's token
    // strings to reconstruct the prompt — or fall back to empty string
    // (the backend will use whatever is in the active session).
    std::string prompt;
    {
        auto bundle = m_impl->view.current.load();
        if (bundle) {
            for (const auto& s : bundle->token_strs)
                prompt += s;
        }
    }

    const int top_k = (kLayers > 0) ? kLayers : 5;
    const std::string ws_url =
        m_impl->ws_base + "/ws/sessions/" + kSessionName + "/logit-lens";

    httplib::ws::WebSocketClient ws(ws_url);
    ws.set_connection_timeout(3, 0);
    ws.set_read_timeout(30, 0);
    ws.set_write_timeout(5, 0);

    if (!ws.is_valid()) {
        m_impl->log(Severity::Warn, "hf",
                    "getLogitLensTrajectory: invalid WS URL: " + ws_url);
        return {};
    }
    if (!ws.connect()) {
        m_impl->log(Severity::Warn, "hf",
                    "getLogitLensTrajectory: WS connect failed: " + ws_url);
        return {};
    }

    // Send config.
    const json config = {{"prompt", prompt}, {"top_k", top_k}};
    if (!ws.send(config.dump())) {
        m_impl->log(Severity::Warn, "hf",
                    "getLogitLensTrajectory: WS send failed");
        ws.close();
        return {};
    }

    std::vector<LogitLensRow> rows;
    std::string msg;
    while (true) {
        const auto rr = ws.read(msg);
        if (rr == httplib::ws::ReadResult::Fail) break;
        if (rr != httplib::ws::ReadResult::Text) continue;

        json frame;
        try { frame = json::parse(msg); }
        catch (...) { continue; }

        const std::string type = json_string(frame, "type");
        if (type == "complete" || type == "error") break;
        if (type != "data") continue;

        // Each "data" frame has a "predictions" array: one entry per
        // token position.  We extract position `token` (clamp to valid).
        const int layer = json_int(frame, "layer", static_cast<int>(rows.size()));

        LogitLensRow row;
        row.layer       = layer;
        row.p1          = kNoFloat;
        row.p2          = kNoFloat;
        row.entropy     = kNoFloat;
        row.is_resolved = false;

        auto preds_it = frame.find("predictions");
        if (preds_it != frame.end() && preds_it->is_array()) {
            const auto& positions = *preds_it;
            const int pos = std::clamp(token, 0,
                                       static_cast<int>(positions.size()) - 1);
            if (pos >= 0 && pos < static_cast<int>(positions.size())) {
                const auto& top = positions[static_cast<std::size_t>(pos)];
                if (top.is_array() && !top.empty()) {
                    const auto& first = top[0];
                    row.top1 = first.value("token", std::string{});
                    row.p1   = static_cast<float>(first.value("prob", 0.0));
                    if (top.size() > 1) {
                        const auto& second = top[1];
                        row.top2 = second.value("token", std::string{});
                        row.p2   = static_cast<float>(second.value("prob", 0.0));
                    }
                }
            }
        }

        // Extract entropy from the per-layer metrics array if present.
        auto metrics_it = frame.find("metrics");
        if (metrics_it != frame.end() && metrics_it->is_array()) {
            for (const auto& m : *metrics_it) {
                const std::string name = m.value("name", std::string{});
                if (name == "entropy" || name == "H") {
                    const auto& vals = m.find("values");
                    if (vals != m.end() && vals->is_array()) {
                        const int pos = std::clamp(token, 0,
                            static_cast<int>(vals->size()) - 1);
                        if (pos >= 0 && pos < static_cast<int>(vals->size())) {
                            row.entropy = static_cast<float>((*vals)[static_cast<std::size_t>(pos)]);
                        }
                    }
                }
            }
        }

        // Mark as resolved once top-1 stabilises (same token as previous
        // layer, if any).
        if (!rows.empty() && rows.back().top1 == row.top1 && !row.top1.empty()) {
            row.is_resolved = true;
        }

        rows.push_back(std::move(row));
    }

    ws.close();

    // Capability update: we have logit-lens data.
    if (!rows.empty()) {
        std::lock_guard<std::mutex> lk(m_impl->mu);
        m_impl->view.capabilities.has_logit_lens = true;
    }

    return rows;
}

// getOutputLogits: open WS /ws/sessions/{n}/generate, stream tokens,
// collect top-k from each frame.  Return the last frame's top-k as
// LogitDist vector.  Returns {} on WS failure or no frames.
std::vector<LogitDist> HFProxyEngine::getOutputLogits(int k)
{
    std::string prompt;
    {
        auto bundle = m_impl->view.current.load();
        if (bundle) {
            for (const auto& s : bundle->token_strs)
                prompt += s;
        }
    }

    const int display_top_k = (k > 0) ? k : 5;
    const std::string ws_url =
        m_impl->ws_base + "/ws/sessions/" + kSessionName + "/generate";

    httplib::ws::WebSocketClient ws(ws_url);
    ws.set_connection_timeout(3, 0);
    ws.set_read_timeout(120, 0);   // generation can be slow
    ws.set_write_timeout(5, 0);

    if (!ws.is_valid()) {
        m_impl->log(Severity::Warn, "hf",
                    "getOutputLogits: invalid WS URL: " + ws_url);
        return {};
    }
    if (!ws.connect()) {
        m_impl->log(Severity::Warn, "hf",
                    "getOutputLogits: WS connect failed: " + ws_url);
        return {};
    }

    // Send generation config.
    const json config = {
        {"prompt",        prompt},
        {"max_tokens",    64},
        {"temperature",   1.0},
        {"display_top_k", display_top_k},
    };
    if (!ws.send(config.dump())) {
        m_impl->log(Severity::Warn, "hf",
                    "getOutputLogits: WS send failed");
        ws.close();
        return {};
    }

    // Accumulate token_ids / token_strs from each "data" frame.
    // Also track the last frame's top_k for the return value.
    std::vector<TokenId>     new_token_ids;
    std::vector<std::string> new_token_strs;
    std::vector<LogitDist>   last_top_k;

    std::string msg;
    while (true) {
        const auto rr = ws.read(msg);
        if (rr == httplib::ws::ReadResult::Fail) break;
        if (rr != httplib::ws::ReadResult::Text) continue;

        json frame;
        try { frame = json::parse(msg); }
        catch (...) { continue; }

        const std::string type = json_string(frame, "type");
        if (type == "error") {
            m_impl->log(Severity::Warn, "hf",
                        "getOutputLogits: backend error: " + json_string(frame, "message"));
            break;
        }
        if (type == "complete") break;
        if (type != "data") continue;

        const std::string tok_str = json_string(frame, "token");
        const int         tok_id  = json_int(frame, "token_id", 0);
        new_token_ids .push_back(static_cast<TokenId>(tok_id));
        new_token_strs.push_back(tok_str);

        // Parse top_k for this step.
        auto top_it = frame.find("top_k");
        if (top_it != frame.end() && top_it->is_array()) {
            last_top_k.clear();
            last_top_k.reserve(top_it->size());
            for (const auto& entry : *top_it) {
                LogitDist ld;
                ld.token    = entry.value("token", std::string{});
                ld.prob     = static_cast<float>(entry.value("prob", 0.0));
                ld.delta    = 0.0f;
                ld.selected = false;
                last_top_k.push_back(std::move(ld));
            }
        }
    }

    ws.close();

    // Extend the current bundle with the generated tokens.
    if (!new_token_strs.empty()) {
        auto bundle = m_impl->view.current.load();
        if (bundle) {
            // Allocate a new bundle that extends the current one.
            auto extended = std::make_shared<CaptureBundle>(*bundle);
            for (std::size_t i = 0; i < new_token_ids.size(); ++i) {
                extended->token_ids .push_back(new_token_ids[i]);
                extended->token_strs.push_back(new_token_strs[i]);
            }
            {
                std::lock_guard<std::mutex> lk(m_impl->mu);
                m_impl->view.captures[extended->prompt_hash] = extended;
                m_impl->view.capabilities.has_token_stream = true;
            }
            m_impl->view.current.store(extended);
        }
    }

    // Mark top-1 as selected if there is a top-1.
    if (!last_top_k.empty()) {
        last_top_k[0].selected = true;
    }

    return last_top_k;
}

}  // namespace llmengine
