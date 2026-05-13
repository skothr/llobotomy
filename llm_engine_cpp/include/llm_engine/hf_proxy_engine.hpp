#pragma once
//
// HFProxyEngine — Model implementation that talks to the FastAPI backend at
// testing/gui/backend/.  See docs/HFPROXY_PLAN.md for the endpoint mapping.
//
// Phase 1 scope (this file):
//   - loadCheckpoint(model_id)   → POST   /api/sessions {name, model_id, mode}
//   - unloadCheckpoint()         → DELETE /api/sessions/{name}
//   - drainEngineLogs()          → flush queued log lines (HTTP errors etc.)
//
// Every other Model::* method falls through to MockModel's default — which
// returns the no-data sentinel for the type when LLOB_USE_MOCK_DATA=OFF
// (release / real-backend builds), or deterministic mock data when ON
// (development builds).  The latter is useful for incrementally validating
// the wiring: load a real model and confirm topology populates while
// per-frame samplers continue to show familiar mock data.
//
// Threading: every method here runs on the UI thread.  loadCheckpoint may
// block for the duration of the backend's model load (10s of seconds for
// a multi-billion-parameter model) — Phase 2 introduces a worker thread
// + async load contract per ENGINE_API.md §2.2.
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

class HFProxyEngine : public MockModel {
public:
    // base_url example: "http://127.0.0.1:8000".  No trailing slash.
    explicit HFProxyEngine(std::string base_url);
    ~HFProxyEngine() override;

    HFProxyEngine(const HFProxyEngine&)            = delete;
    HFProxyEngine& operator=(const HFProxyEngine&) = delete;

    // ── Wired hooks ───────────────────────────────────────────────────────
    CheckpointResult       loadCheckpoint  (std::string_view path) override;
    void                   unloadCheckpoint()                      override;
    std::vector<LogEntry>  drainEngineLogs ()                      override;
    ModelInfo              getModelInfo    ()                      override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace llmengine
