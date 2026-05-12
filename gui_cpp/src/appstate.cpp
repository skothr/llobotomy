#include "appstate.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
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

// ── Mutators ───────────────────────────────────────────────────────────────

void AppState::toggleProbe(int L, int h) {
    char buf[24];
    std::snprintf(buf, sizeof buf, "%d.%d", L, h);
    const bool on = toggle(probedHeads, std::string(buf));
    char msg[64];
    std::snprintf(msg, sizeof msg, "%s probe @ block_%d.head_%d",
                  on ? "attached" : "detached", L, h);
    pushLog("probe", msg);
}

void AppState::toggleAblate(std::string_view headKey) {
    const bool on = toggle(ablatedHeads, std::string(headKey));
    pushLog("ablate", std::string(on ? "ablated head " : "restored head ") + std::string(headKey));
}

void AppState::toggleProbeComp(int L, std::string_view comp) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%d.%.40s", L, std::string(comp).c_str());
    const bool on = toggle(probedComponents, std::string(buf));
    pushLog("probe", std::string(on ? "attached probe @ " : "detached probe @ ") + buf);
}

void AppState::toggleAblateComp(int L, std::string_view comp) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%d.%.40s", L, std::string(comp).c_str());
    const bool on = toggle(ablatedComponents, std::string(buf));
    pushLog("ablate", std::string(on ? "ablated " : "restored ") + buf);
}

void AppState::toggleLayerExpand(int L) {
    toggle(expandedLayers, L);
}

void AppState::toggleLayerSkip(int L) {
    const bool on = toggle(skippedLayers, L);
    char msg[64];
    std::snprintf(msg, sizeof msg, "%s block_%d (whole-layer bypass)",
                  on ? "skipped" : "restored", L);
    pushLog("ablate", msg);
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

void AppState::pushLog(std::string kind, std::string msg) {
    while (logs.size() >= 500) logs.pop_front();
    logs.push_back(LogEntry{ NowMs(), std::move(kind), std::move(msg) });
}

// ── Demo bootstrap (matches app.jsx defaults) ─────────────────────────────

void AppState::seedDemo() {
    model = ModelInfo{
        .name = "tiny-decoder",
        .nLayers = 6, .nHeads = 6, .dModel = 384,
        .dHead = 64, .dMlp = 1536, .vocab = 32000,
    };
    // Default layer for the 6-layer tiny model.
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

    // Pre-seeded ablations / probes (match the screenshot state).
    ablatedHeads.insert("4.3");
    ablatedHeads.insert("4.5");
    ablatedHeads.insert("2.2");
    probedHeads.insert("4.3");
    probedHeads.insert("5.1");
    probedComponents.insert("4.W_Q");

    // Seed a starter log feed.
    const std::int64_t now = NowMs();
    logs.push_back({now - 9000, "init",   "llobotomy 0.4.18-rc2 starting"});
    logs.push_back({now - 8800, "init",   "cuda:0 detected · A100 80GB"});
    logs.push_back({now - 8500, "load",   "loaded checkpoint tiny-decoder (548MB)"});
    logs.push_back({now - 8000, "hook",   "registered 144 forward hooks"});
    logs.push_back({now - 7400, "probe",  "loaded probe set: refusal_v2 (14 active)"});
    logs.push_back({now - 6800, "sae",    "mounted SAE L04-mlp (16384 features)"});
    logs.push_back({now - 6000, "fwd",    "forward pass · L4 · 12.3ms"});
    logs.push_back({now - 4200, "ablate", "ablated head 4.3 · zero"});
    logs.push_back({now - 3800, "ablate", "ablated head 4.5 · zero"});
    logs.push_back({now - 3400, "ablate", "ablated head 2.2 · zero"});
    logs.push_back({now - 2200, "probe",  "attached probe @ block_4.head_3"});
    logs.push_back({now - 1800, "probe",  "attached probe @ block_5.head_1"});
    logs.push_back({now -  800, "fwd",    "forward pass · L4 · 11.7ms"});
    lastSyntheticLog = std::chrono::steady_clock::now();
    lastStepTime     = std::chrono::steady_clock::now();
}

// ── Live tick: synthetic logs + inference step ────────────────────────────

void AppState::tickLiveFeed() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();

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

    // Synthetic logs (~1200ms cadence) when liveAnim + liveLogs.
    if (liveAnim && liveLogs) {
        const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSyntheticLog).count();
        if (dt >= 1200) {
            static thread_local std::mt19937 rng{0xC0DEBA5Eu};
            std::uniform_int_distribution<int> samp(0, 3);
            std::uniform_int_distribution<int> Ld(0, std::max(0, model.nLayers - 1));
            std::uniform_real_distribution<float> ms(8.0f, 12.0f);
            char buf[96];
            const int kind = samp(rng);
            const int L    = Ld(rng);
            switch (kind) {
                case 0: std::snprintf(buf, sizeof buf, "forward pass · L%d · %.2fms", L, ms(rng)); pushLog("fwd",   buf); break;
                case 1: std::snprintf(buf, sizeof buf, "hook fired · resid_post @ L%d", L);        pushLog("hook",  buf); break;
                case 2: std::snprintf(buf, sizeof buf, "KV cache · %d tok · %.2fMB", 128 + L * 16, 0.3f + ms(rng) * 0.04f);
                                                                                                     pushLog("cache", buf); break;
                default: std::snprintf(buf, sizeof buf, "vram delta · %+dMB", (L * 7) - 30);        pushLog("mem",   buf); break;
            }
            lastSyntheticLog = now;
        }
    }
}

}  // namespace llob
