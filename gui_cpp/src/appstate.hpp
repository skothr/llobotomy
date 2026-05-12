#pragma once
#include "style.hpp"

#include <imgui.h>

#include <chrono>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace llob {

struct Model;     // fwd; src/model/model.hpp

struct ModelInfo {
    std::string name;
    int nLayers;
    int nHeads;
    int dModel;
    int dHead;
    int dMlp;
    int vocab;
};

enum class Workspace : int {
    Arch = 0, Inf, Attn, Probes, Train, Ft, Data, Raw, Logs, Count
};

struct LogEntry {
    std::int64_t  ts_ms;     // unix ms
    std::string   kind;      // fwd/hook/cache/mem/probe/ablate/...
    std::string   msg;
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

    // Inference run loop
    bool                      running        = false;
    int                       stepIdx        = 11;
    std::chrono::steady_clock::time_point lastStepTime{};

    // Logs (capped at 500)
    std::deque<LogEntry>      logs;
    bool                      liveLogs       = true;
    std::chrono::steady_clock::time_point lastSyntheticLog{};

    // Theme / preferences
    Theme                     theme          = Theme::Tracy;
    Density                   density        = Density::Normal;
    ImU32                     accentOverride = 0;     // 0 = use theme default
    bool                      showRaw        = true;
    bool                      liveAnim       = true;

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

    void  pushLog(std::string kind, std::string msg);

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
