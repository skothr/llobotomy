#pragma once
#include "keybindings.hpp"
#include "style.hpp"

#include "llm_engine/hf_proxy_engine.hpp"
#include "llm_engine/log.hpp"
#include "llm_engine/model.hpp"

#include <imgui.h>

#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace llob {

// Engine types live in `llmengine::` (testing/llm_engine_cpp/).  Re-export
// the surface gui_cpp consumes into `llob::` so existing call sites
// (`llob::Model`, `llob::Severity`, the various DTOs) keep resolving
// without a sweep across every workspace.  New code outside this file is
// free to use the canonical `llmengine::` names directly.
using llmengine::Severity;
using llmengine::LogEntry;
using llmengine::SeverityName;
using llmengine::SeverityShort;
using llmengine::Model;
using llmengine::MockModel;
using llmengine::HFProxyEngine;
using llmengine::HeadBias;
using llmengine::Mulberry32;
using llmengine::kNoFloat;
using llmengine::kNoInt;
using llmengine::kNoSize;
// Architecture / weights
using llmengine::ParamBreakdownRow;
using llmengine::LiveActivations;
using llmengine::TensorMeta;
using llmengine::TensorStats;
using llmengine::DiffStats;
// Inference
using llmengine::ResidualContribution;
using llmengine::ResidualSummary;
using llmengine::LogitLensRow;
using llmengine::LogitDist;
using llmengine::MlpFeatureActivation;
using llmengine::ProbeEntry;
using llmengine::SteeringConfig;
// Attention
using llmengine::QKVStats;
using llmengine::HeadStatRow;
using llmengine::PatchSourceState;
// Probes / SAE
using llmengine::FeatureSummary;
using llmengine::FeatureCard;
using llmengine::FeatureExample;
using llmengine::CoFiringEntry;
using llmengine::SAETrainingMetrics;
using llmengine::ProbeTrainState;
// Training
using llmengine::TrainingState;
using llmengine::TrainingMetricCard;
using llmengine::LossCurve;
// Finetune
using llmengine::LoRAConfig;
using llmengine::OptimizerConfig;
using llmengine::DataConfig;
using llmengine::EvalDiffMetric;
using llmengine::ABSample;
// Datasets
using llmengine::DatasetSummary;
using llmengine::DatasetSpan;
using llmengine::DatasetSample;
using llmengine::DatasetSampleStats;
using llmengine::SourceMixRow;
using llmengine::DatasetDistribution;
// Engine runtime
using llmengine::EngineMetrics;
using llmengine::ExportEntry;
// Topology — promoted from the engine; gui_cpp's previous local
// `struct ModelInfo` was a strict subset of llmengine::ModelInfo.
using llmengine::ModelInfo;

enum class Workspace : int {
    Arch = 0, Inf, Attn, Probes, Train, Ft, Data, Raw, Logs, Count
};

struct ProjectTab {
    std::string id;
    std::string name;
    enum class Dot { Accent, Warn, Bad, Dim } dot;
    std::string flag;        // optional ("live", etc.)
};

// Single source of truth shared across all workspaces.
struct AppState {
    // Workspace & projects
    Workspace                 activeWs       = Workspace::Arch;
    std::vector<ProjectTab>   projects;
    std::string               activeProject;

    // Model (mockable — see model.hpp)
    ModelInfo                 model{};
    std::vector<std::string>  sampleTokens;     // used by inference/attn
    // Path the active checkpoint was loaded from (set by File ▸ Open and
    // by loadCheckpoint).  Empty when no checkpoint is open.  Drives the
    // per-checkpoint sidecar JSON location.
    std::string               checkpointPath;

    // Global selection (zero / empty when no model is loaded — the
    // demo seeder overrides these with sensible defaults for its model).
    int                       activeLayer    = 0;
    int                       activeToken    = 0;
    int                       activeHead     = 0;
    std::string               activeTensor   = "";

    // Ablation / probe sets — keys "L.h" or "L.<comp>"
    std::unordered_set<std::string> ablatedHeads;
    std::unordered_set<std::string> probedHeads;
    std::unordered_set<std::string> ablatedComponents;
    std::unordered_set<std::string> probedComponents;

    // Architecture-map per-layer state
    std::unordered_set<int>   expandedLayers;
    std::unordered_set<int>   skippedLayers;
    float                     archZoom       = 1.0f;
    // One-shot flag — set by Ctrl+0 or the (+) fit button; consumed by
    // DrawArchMap on the next frame (it has the geometry info needed to
    // compute the fit ratio against the current viewport).
    bool                      archRequestFit = false;

    // Inference run loop
    bool                      running        = false;
    int                       stepIdx        = 11;
    std::chrono::steady_clock::time_point lastStepTime{};

    // Prompt input — shared across inference, attention, and probes
    // workspaces via the reusable DrawPromptInput widget.  `promptDraft`
    // is the live editable buffer; `lastSubmittedPrompt` is the most
    // recent string sent to Model::setActivePrompt (used to detect
    // "submitted but unchanged" vs "fresh submit").  `promptSubmittedAt`
    // helps UI show "submitted Xs ago" affordances.
    std::string                                   promptDraft;
    std::string                                   lastSubmittedPrompt;
    std::chrono::steady_clock::time_point         promptSubmittedAt{};

    // Logs — capped at 500 in memory; full history goes to a log file
    // (path returned by Logger::path()).  Reads from the UI thread; writes
    // protected by logs_mu so background threads (engine bridge, future
    // stderr capture) can push safely.
    std::deque<LogEntry>      logs;
    mutable std::mutex        logs_mu;
    bool                      liveLogs       = true;     // tail toggle in logs UI

    // Theme / preferences
    Theme                     theme          = Theme::Tracy;
    Density                   density        = Density::Normal;
    ImU32                     accentOverride = 0;     // 0 = use theme default
    bool                      showRaw        = true;
    bool                      liveAnim       = true;
    // User-configurable keybindings — defaults populated by Defaults();
    // overlaid by Load() on startup; persisted on every rebind.
    Keybindings               bindings       = Keybindings::Defaults();

    // Render-time stats (updated each frame from main loop)
    float                     renderMs       = 14.2f;
    float                     fps            = 60.0f;

    // ─── Mutators (mirror app.jsx) ────────────────────────────────────────
    void  toggleProbe(int L, int h);
    void  toggleAblate(std::string_view headKey);   // key = "L.h"
    void  toggleProbeComp(int L, std::string_view comp);
    void  toggleAblateComp(int L, std::string_view comp);
    void  toggleLayerExpand(int L);
    void  toggleLayerSkip(int L);

    void  setActiveLayer(int L);
    void  setActiveToken(int T);
    void  setActiveHead(int H);
    void  setActiveTensor(std::string name);

    // Primary log entry point — thread-safe.  Fans out to the in-memory
    // ring + the on-disk log file (opened via Logger::init in main.cpp).
    void  pushLog(Severity sev, std::string kind, std::string msg);
    // Convenience overload — defaults to Severity::Info.  Kept for the
    // common case of UI event logging.
    void  pushLog(std::string kind, std::string msg);

    // Snapshot of the most recent N entries — UI rendering grabs this so
    // the lock is held briefly.
    std::vector<LogEntry> snapshotLogs(std::size_t n_tail) const;

    // Per-session UI defaults (theme already at sane defaults; this only
    // touches non-engine state so it's safe to always call on startup).
    void  seedSession();

    // Demo-only seeding — populates a 6-layer toy model, sample tokens,
    // pre-set ablations / probes, project tabs, and a few canned log
    // entries.  Gated by LLOB_USE_MOCK_DATA at the call site so a real
    // build doesn't get pre-populated with fake state.
    void  seedMockData();

    // True when a Model has been wired up (nLayers > 0 etc.) — UI uses
    // this to decide whether to render workspace contents or a "no model
    // loaded" placeholder.
    bool  hasModel() const { return model.nLayers > 0; }

    // Pull non-mock state from a real Model (model topology + current
    // tokenization, when the engine exposes them).
    void  loadFromModel(Model& m);

    void  tickLiveFeed();          // synthetic logs + run-loop stepping
};

// Workspace metadata — id, label with prefix glyph.
struct WorkspaceDef {
    Workspace     id;
    const char*   key;             // "arch", "inf", ...
    const char*   label;           // "◫ architecture"
    const char*   short_label;     // "arch"
};
const WorkspaceDef& WsDef(Workspace w);
constexpr int kNumWorkspaces = static_cast<int>(Workspace::Count);

}  // namespace llob
