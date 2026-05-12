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
    std::string               activeProject  = "p1";

    // Model (mockable — see model.hpp)
    ModelInfo                 model{};
    std::vector<std::string>  sampleTokens;     // used by inference/attn

    // Global selection
    int                       activeLayer    = 8;
    int                       activeToken    = 11;
    int                       activeHead     = 3;
    std::string               activeTensor   = "blocks.8.attn.W_Q.weight";

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
    void  seedDemo();              // initial projects, ablations, sample logs
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
