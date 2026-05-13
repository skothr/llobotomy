#include "appstate.hpp"

#include "logger.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <utility>

namespace llob {

namespace {

constexpr WorkspaceDef kWorkspaces[kNumWorkspaces] = {
    { Workspace::Arch,   "arch",   "◫ architecture",  "arch"   },
    { Workspace::Inf,    "inf",    "▶ inference",     "inf"    },
    { Workspace::Attn,   "attn",   "▦ attention",     "attn"   },
    { Workspace::Probes, "probes", "◈ probes/SAE",    "probes" },
    { Workspace::Train,  "train",  "∿ training",      "train"  },
    { Workspace::Ft,     "ft",     "◆ finetune",      "ft"     },
    { Workspace::Data,   "data",   "≡ datasets",      "data"   },
    { Workspace::Raw,    "raw",    "0x raw_tensors",  "raw"    },
    { Workspace::Logs,   "logs",   "█ logs",          "logs"   },
};

std::int64_t NowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

template <typename Set, typename Key>
bool toggle(Set& set, Key&& key) {
    auto it = set.find(key);
    if (it == set.end()) { set.emplace(std::forward<Key>(key)); return true; }
    set.erase(it); return false;
}

}  // namespace

const WorkspaceDef& WsDef(Workspace w) { return kWorkspaces[static_cast<int>(w)]; }

// SeverityName / SeverityShort moved to llm_engine_cpp/src/log.cpp — they
// belong to the engine's log primitives, not the GUI state.

// ── Mutators ───────────────────────────────────────────────────────────────

void AppState::toggleProbe(int L, int h) {
    char buf[24];
    std::snprintf(buf, sizeof buf, "%d.%d", L, h);
    const bool on = toggle(probedHeads, std::string(buf));
    LLOB_LOG_INFO("probe", "%s probe @ block_%d.head_%d",
                  on ? "attached" : "detached", L, h);
}

void AppState::toggleAblate(std::string_view headKey) {
    const bool on = toggle(ablatedHeads, std::string(headKey));
    LLOB_LOG_INFO("ablate", "%s head %.*s",
                  on ? "ablated" : "restored",
                  static_cast<int>(headKey.size()), headKey.data());
}

void AppState::toggleProbeComp(int L, std::string_view comp) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%d.%.40s", L, std::string(comp).c_str());
    const bool on = toggle(probedComponents, std::string(buf));
    LLOB_LOG_INFO("probe", "%s probe @ %s",
                  on ? "attached" : "detached", buf);
}

void AppState::toggleAblateComp(int L, std::string_view comp) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%d.%.40s", L, std::string(comp).c_str());
    const bool on = toggle(ablatedComponents, std::string(buf));
    LLOB_LOG_INFO("ablate", "%s %s", on ? "ablated" : "restored", buf);
}

void AppState::toggleLayerExpand(int L) {
    toggle(expandedLayers, L);
}

void AppState::toggleLayerSkip(int L) {
    const bool on = toggle(skippedLayers, L);
    LLOB_LOG_INFO("ablate", "%s block_%d (whole-layer bypass)",
                  on ? "skipped" : "restored", L);
}

void AppState::setActiveLayer(int L) {
    activeLayer = std::clamp(L, 0, std::max(0, model.nLayers - 1));
}
void AppState::setActiveToken(int T) {
    const int n = static_cast<int>(sampleTokens.size());
    activeToken = std::clamp(T, 0, std::max(0, n - 1));
}
void AppState::setActiveHead(int H) {
    activeHead = std::clamp(H, 0, std::max(0, model.nHeads - 1));
}
void AppState::setActiveTensor(std::string name) { activeTensor = std::move(name); }

void AppState::pushLog(Severity sev, std::string kind, std::string msg) {
    std::lock_guard<std::mutex> lk(logs_mu);
    while (logs.size() >= 500) logs.pop_front();
    logs.push_back(LogEntry{ NowMs(), sev, std::move(kind), std::move(msg) });
}

void AppState::pushLog(std::string kind, std::string msg) {
    pushLog(Severity::Info, std::move(kind), std::move(msg));
}

std::vector<LogEntry> AppState::snapshotLogs(std::size_t n_tail) const {
    std::lock_guard<std::mutex> lk(logs_mu);
    if (n_tail == 0 || logs.empty()) return {};
    const std::size_t n = std::min(n_tail, logs.size());
    return std::vector<LogEntry>(logs.end() - static_cast<std::ptrdiff_t>(n), logs.end());
}

// ── Session bootstrap ─────────────────────────────────────────────────────

void AppState::seedSession() {
    // Defaults that aren't engine-data: theme, density, layout flags.  The
    // struct itself already has these in member initialisers, so this is
    // currently a no-op — the function exists as the seam where session
    // prefs would be loaded from disk in a future polish pass.
    lastStepTime     = std::chrono::steady_clock::now();
}

void AppState::loadFromModel(Model& /*m*/) {
    // [DATA HOOK] Once Model exposes getModelInfo() / getCurrentTokens(),
    // populate `model` and `sampleTokens` from the engine here.  The
    // current Model interface (model.hpp) doesn't have those yet — they
    // belong on the next round of hooks.
}

// ── Demo seed (mock-only) ────────────────────────────────────────────────

void AppState::seedMockData() {
    model = ModelInfo{
        .name = "tiny-decoder",
        .nLayers = 6, .nHeads = 6, .dModel = 384,
        .dHead = 64, .dMlp = 1536, .vocab = 32000,
    };
    activeLayer = 4;
    activeHead  = 3;
    expandedLayers = { activeLayer };

    sampleTokens = {
        "<bos>", "▁When", "▁the", "▁transformer",
        "▁processes", "▁a", "▁sentence", ",",
        "▁each", "▁attention", "▁head",
        "▁attends", "▁to", "▁a",
        "▁subset", "▁of", "▁the",
        "▁previous", "▁tokens", ".",
    };
    activeToken = 11;
    stepIdx     = 11;

    projects = {
        { "p1", "inv/attends_head_L4", ProjectTab::Dot::Accent, "live" },
        { "p2", "sae/L04-mlp_run3",    ProjectTab::Dot::Warn,   "" },
        { "p3", "lora/instruct_v3",    ProjectTab::Dot::Warn,   "" },
        { "p4", "sandbox",             ProjectTab::Dot::Dim,    "" },
    };
    activeProject = "p1";

    ablatedHeads.insert("4.3");
    ablatedHeads.insert("4.5");
    ablatedHeads.insert("2.2");
    probedHeads.insert("4.3");
    probedHeads.insert("5.1");
    probedComponents.insert("4.W_Q");

    // Mock-mode demo logs are emitted via the normal logger so they hit
    // both the in-memory ring AND the on-disk log — same pipeline real
    // events go through.  Tagged "demo" so they're easy to filter out.
    LLOB_LOG_INFO("demo", "demo mode: pre-loaded toy model + ablations");
}

// ── Live tick: synthetic logs + inference step ────────────────────────────

void AppState::tickLiveFeed() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();

    // No model loaded → nothing to step.
    if (!hasModel() || sampleTokens.empty()) return;

    // Run-loop step (~380ms per token).
    if (running) {
        const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastStepTime).count();
        if (dt >= 380) {
            const int n = static_cast<int>(sampleTokens.size());
            int next = stepIdx + 1;
            if (next >= n) { running = false; next = n - 1; }
            stepIdx     = next;
            activeToken = next;
            lastStepTime = now;
        }
    }
}

}  // namespace llob
