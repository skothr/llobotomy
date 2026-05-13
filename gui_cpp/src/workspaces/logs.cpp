// Logs workspace — log stream | severity/source filters | live counters.
// HANDOFF §3.9.
//
// Source of truth is AppState.logs (snapshotted under its own mutex so a
// background-thread engine push doesn't tear the read).  Filters apply on
// the rendered slice; the underlying ring keeps everything for the on-disk
// log file.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "logger.hpp"
#include "llm_engine/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/fmt.hpp"

#include <imgui.h>
#include <imgui_internal.h>           // DockBuilder*

#include <chrono>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>

namespace llob {

namespace {

ImU32 SevColor(Severity s) {
    switch (s) {
        case Severity::Trace: return Sty().text_dim;
        case Severity::Debug: return Sty().text_muted;
        case Severity::Info:  return Sty().info;
        case Severity::Warn:  return Sty().warn;
        case Severity::Error: return Sty().bad;
        case Severity::Fatal: return Sty().bad;
    }
    return Sty().text_muted;
}

ImU32 KindColor(const std::string& kind) {
    if (kind == "ablate")               return Sty().bad;
    if (kind == "probe" || kind == "export" || kind == "run") return Sty().accent;
    if (kind == "fwd"   || kind == "hook")                    return Sty().info;
    if (kind == "train")                                      return Sty().good;
    if (kind == "glfw"  || kind == "font" || kind == "imgui") return Sty().warn;
    return Sty().text_muted;
}

// Filter state lives at module scope so it persists across the workspace's
// re-creation each frame.  Severity defaults: everything visible except
// trace; user can flip in the UI.
struct Filters {
    bool sev[6] = { /*trace*/ false, /*debug*/ true, /*info*/ true,
                    /*warn*/ true,  /*error*/ true, /*fatal*/ true };
    std::unordered_map<std::string, bool> src;   // kind → enabled (default: enabled)
    char search[64] = {};
};
Filters& filters() { static Filters f; return f; }

bool PassesFilter(const LogEntry& e) {
    auto& f = filters();
    if (!f.sev[static_cast<int>(e.sev)]) return false;
    auto it = f.src.find(e.kind);
    if (it != f.src.end() && !it->second) return false;
    if (f.search[0]) {
        // Case-insensitive substring match against msg + kind.
        auto haystack = e.kind + " " + e.msg;
        std::string needle = f.search;
        for (auto& c : haystack) c = static_cast<char>(std::tolower(c));
        for (auto& c : needle)   c = static_cast<char>(std::tolower(c));
        if (haystack.find(needle) == std::string::npos) return false;
    }
    return true;
}

void DrawLogStream(AppState& s) {
    // Snapshot the full ring once; filter at render time.
    const auto entries = s.snapshotLogs(500);
    std::size_t shown = 0;
    for (const auto& e : entries) if (PassesFilter(e)) ++shown;

    char flag[80];
    std::snprintf(flag, sizeof flag, "%zu / %zu entries · tail %s · %s",
                  shown, entries.size(),
                  s.liveLogs ? "ON" : "OFF",
                  LoggerPath().empty() ? "no log file"
                                         : ("file: " + LoggerPath()).c_str());
    DrawTitleBar("logstream", "█", flag, "logstream", [&] {
        if (ImGui::SmallButton(s.liveLogs ? "|| tail" : "> tail")) s.liveLogs = !s.liveLogs;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(150);
        ImGui::InputTextWithHint("##s", "search…", filters().search,
                                  sizeof filters().search);
    });
    if (!ImGui::BeginChild("##log_body", ImVec2(0, 0), ImGuiChildFlags_None,
                            ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGui::EndChild(); return;
    }
    if (entries.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted("// no log entries yet");
        ImGui::PopStyleColor();
        ImGui::EndChild(); return;
    }
    for (const auto& l : entries) {
        if (!PassesFilter(l)) continue;
        const auto secs = (l.ts_ms / 1000) % 86400;
        const int hh = int(secs / 3600);
        const int mm = int((secs / 60) % 60);
        const int ss = int(secs % 60);
        const int cs = int((l.ts_ms % 1000) / 10);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::Text("%02d:%02d:%02d.%02d", hh, mm, ss, cs);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, SevColor(l.sev));
        ImGui::Text("%s", SeverityShort(l.sev));
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, KindColor(l.kind));
        ImGui::Text("[%s]", l.kind.c_str());
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextUnformatted(l.msg.c_str());
    }
    if (s.liveLogs) {
        const auto blink = (std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count() / 500) % 2;
        if (blink == 0) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
            ImGui::TextUnformatted("█"); ImGui::PopStyleColor();
        }
    }
    if (s.liveLogs)
        ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
}

void DrawFilters(AppState& s) {
    DrawTitleBar("filters", "⚲", nullptr, "filters");
    if (!ImGui::BeginChild("##fl_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }

    if (auto sec = BeginSection("severity", true)) {
        const char* names[6] = {"trace","debug","info","warn","error","fatal"};
        for (int i = 0; i < 6; ++i) {
            ImGui::PushStyleColor(ImGuiCol_CheckMark, SevColor(static_cast<Severity>(i)));
            ImGui::Checkbox(names[i], &filters().sev[i]);
            ImGui::PopStyleColor();
        }
    }

    // Source list is built dynamically from the kinds present in the
    // current log ring.  New sources appear automatically (default enabled).
    std::set<std::string> kinds_seen;
    for (const auto& e : s.snapshotLogs(500)) kinds_seen.insert(e.kind);
    if (auto sec = BeginSection("source", true)) {
        if (kinds_seen.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
            ImGui::TextUnformatted("// no sources yet");
            ImGui::PopStyleColor();
        }
        for (const auto& k : kinds_seen) {
            // Default each newly-seen kind to enabled.
            auto [it, inserted] = filters().src.try_emplace(k, true);
            ImGui::PushStyleColor(ImGuiCol_CheckMark, KindColor(k));
            ImGui::Checkbox(k.c_str(), &it->second);
            ImGui::PopStyleColor();
        }
    }
    ImGui::EndChild();
}

void DrawMetrics(const AppState& s, Model& m) {
    DrawTitleBar("metrics", "∿", nullptr, "metrics-panel");
    if (!ImGui::BeginChild("##mt_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("Live counters", true)) {
        // Severity rates — counted from the in-memory ring (last 500 entries).
        const auto recent = s.snapshotLogs(500);
        int n_warn = 0, n_err = 0;
        for (const auto& e : recent) {
            if (e.sev == Severity::Warn)  ++n_warn;
            if (e.sev == Severity::Error || e.sev == Severity::Fatal) ++n_err;
        }
        char log_rate[32]; std::snprintf(log_rate, sizeof log_rate, "%zu /min", recent.size() / 60 + 1);
        char warn_r[16];   std::snprintf(warn_r, sizeof warn_r, "%d /min", n_warn);
        char err_r[16];    std::snprintf(err_r,  sizeof err_r,  "%d /min", n_err);
        // [DATA HOOK] Model::getEngineMetrics() — cuda mem, cpu, fwd time
        // come from the engine.  Renders as "—" when unwired.
        const auto em = m.getEngineMetrics();
        char cuda[32];
        if (std::isnan(em.cuda_mem_used_GB) || std::isnan(em.cuda_mem_total_GB))
            std::snprintf(cuda, sizeof cuda, "—");
        else
            std::snprintf(cuda, sizeof cuda, "%.1f / %.1f GB",
                           double(em.cuda_mem_used_GB), double(em.cuda_mem_total_GB));
        char cpu[16];      std::snprintf(cpu,    sizeof cpu,    "%s%%", FmtFloat(em.cpu_pct, "%.1f").c_str());
        // "ms / step" gets truncated in the narrow metrics column at the
        // default dock split — drop the "/ step" suffix (label "fwd time"
        // already implies per-step latency).
        char fwd[24];      std::snprintf(fwd,    sizeof fwd,    "%s ms", FmtFloat(em.fwd_time_ms, "%.1f").c_str());
        KV({
            { "log rate",  log_rate, "accent" },
            { "warn rate", warn_r,   n_warn > 0 ? "warn" : "muted" },
            { "err rate",  err_r,    n_err  > 0 ? "bad"  : "good"  },
            { "cuda mem",  cuda,     "good"   },
            { "cpu",       cpu,      ""       },
            { "fwd time",  fwd,      ""       },
        });
    }
    ImGui::EndChild();
}

}  // namespace

// Layout: log_stream | (filters / metrics)
void BuildLogsLayout(ImGuiID dock_id) {
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, ImGui::GetMainViewport()->Size);

    ImGuiID stream_n, right_n, filters_n, metrics_n;
    ImGui::DockBuilderSplitNode(dock_id, ImGuiDir_Right, 0.22f, &right_n,   &stream_n);
    ImGui::DockBuilderSplitNode(right_n, ImGuiDir_Down,  0.45f, &metrics_n, &filters_n);

    ImGui::DockBuilderDockWindow("logs.stream",  stream_n);
    ImGui::DockBuilderDockWindow("logs.filters", filters_n);
    ImGui::DockBuilderDockWindow("logs.metrics", metrics_n);
    ImGui::DockBuilderFinish(dock_id);
}

void SubmitLogsPanels(AppState& s, Model& m) {
    if (ImGui::Begin("logs.stream",  nullptr, ImGuiWindowFlags_NoTitleBar)) DrawLogStream(s);
    ImGui::End();
    if (ImGui::Begin("logs.filters", nullptr, ImGuiWindowFlags_NoTitleBar)) DrawFilters(s);
    ImGui::End();
    if (ImGui::Begin("logs.metrics", nullptr, ImGuiWindowFlags_NoTitleBar)) DrawMetrics(s, m);
    ImGui::End();
}

}  // namespace llob
