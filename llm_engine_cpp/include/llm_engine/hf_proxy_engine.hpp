#pragma once
//
// HFProxyEngine — Model implementation that talks to the FastAPI backend at
// testing/gui/backend/.  See docs/HFPROXY_PLAN.md for the endpoint mapping.
//
// Phase 1:
//   - loadCheckpoint(model_id)   → POST   /api/sessions {name, model_id, mode}
//   - unloadCheckpoint()         → DELETE /api/sessions/{name}
//   - drainEngineLogs()          → flush queued log lines (HTTP errors etc.)
//
// Phase 2 (this file):
//   - setActivePrompt(prompt)    → POST /api/sessions/{n}/capture (worker thread)
//   - getCurrentTokens()         → view().current token_strs
//   - getAttentionPattern(L,H)   → GET  /api/sessions/{n}/capture/{h}/attention
//   - getResidualSummary(L)      → GET  /api/sessions/{n}/capture/{h}/residual
//   - getQKVStats(L,H,T)         → GET  /api/sessions/{n}/capture/{h}/qkv
//
// Every other Model::* method falls through to Model's default — which
// returns the no-data sentinel for the type (empty vector / NaN / "").
// This is honest: a real backend that hasn't wired a particular getter
// shows an empty panel, never mock data that could be mistaken for real
// output.  Pick LLOB_BACKEND=mock explicitly if you want the screenshot
// data path.
//
// Threading: setActivePrompt pushes a capture job to the SamplerWorker
// thread (already present for heartbeat).  Getters run on the UI thread
// and may issue a single synchronous HTTP fetch (~10ms) for the first
// request of each (layer, head) cell; subsequent calls hit the cache inside
// the CaptureBundle.
//
// Backend selection: see main.cpp's MakeBackend() factory.  LLOB_BACKEND
// picks the implementation; LLOB_BACKEND_URL overrides the FastAPI base
// URL (default http://127.0.0.1:8000).

#include "llm_engine/model.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llmengine {

class HFProxyEngine : public Model {
public:
    // base_url example: "http://127.0.0.1:8000".  No trailing slash.
    explicit HFProxyEngine(std::string base_url);
    ~HFProxyEngine() override;

    HFProxyEngine(const HFProxyEngine&)            = delete;
    HFProxyEngine& operator=(const HFProxyEngine&) = delete;

    // ── Phase 1 hooks ─────────────────────────────────────────────────────
    CheckpointResult       loadCheckpoint  (std::string_view path) override;
    void                   unloadCheckpoint()                      override;
    std::vector<LogEntry>  drainEngineLogs ()                      override;
    ModelInfo              getModelInfo    ()                      override;
    EngineMetrics          getEngineMetrics()                      override;
    const ModelView&       view            () const                override;
    Capabilities           getCapabilities () const                override;

    // ── Phase 2 hooks — real capture data ─────────────────────────────────
    void setActivePrompt(std::string_view prompt)                             override;

    std::vector<std::string>        getCurrentTokens ()                       override;
    std::vector<std::vector<float>> getAttentionPattern(int layer, int head,
                                                         int seqLen,
                                                         HeadBias bias)       override;
    ResidualSummary                 getResidualSummary (int layer)            override;
    QKVStats                        getQKVStats        (int layer, int head,
                                                        int token)            override;

    // ── Phase 3 hooks — intervention wiring ───────────────────────────────
    // setAblation: installs the full ablation set (overwrites any prior
    // pending set). Worker thread diffs against m_view.surgery and issues
    // individual POST /surgery ops + one POST /surgery/commit.
    //
    // setSteering: POST /api/sessions/{n}/steering with {layer, alpha, source}.
    // clearSteering: DELETE /api/sessions/{n}/steering.
    //
    // All three are fire-and-forget from the caller's perspective: they
    // update pending state and wake the worker; the worker does the HTTP
    // round-trip and, on success, updates m_view.surgery.
    void setAblation   (const std::vector<std::string>& head_canonical,
                        const std::vector<std::string>& component_canonical) override;
    void setSteering   (const SteeringConfig& cfg)                           override;
    void clearSteering ()                                                     override;

    // ── Phase 4 hooks — WebSocket streams ─────────────────────────────────
    // getLogitLensTrajectory: opens WS /ws/sessions/{n}/logit-lens, sends
    // {prompt, top_k}, accumulates per-layer frames, returns when "complete"
    // is received or the connection closes.  Synchronous on the caller's
    // thread (intended to be called from a background thread / DerivedCache
    // worker, not the UI thread).
    //
    // getOutputLogits: opens WS /ws/sessions/{n}/generate, streams tokens,
    // returns top-k from the last data frame.  Synchronous similarly.
    //
    // Both override Model's defaulted-empty getters.  On WS connection
    // failure they log a warn and return {}.
    std::vector<LogitLensRow> getLogitLensTrajectory(int token, int kLayers) override;
    std::vector<LogitDist>    getOutputLogits       (int k)                  override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace llmengine
