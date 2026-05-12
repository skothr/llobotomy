// Logs workspace — log stream | severity/source filters | live counters.
// HANDOFF §3.9.

#include "workspaces/workspaces.hpp"

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/widgets.hpp"

#include <imgui.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <unordered_map>

namespace llob {

namespace {

ImU32 SevColor(const std::string& kind) {
    if (kind == "error" || kind == "fatal" || kind == "ablate") return Sty().bad;
    if (kind == "warn")                                          return Sty().warn;
    if (kind == "probe" || kind == "export" || kind == "run")    return Sty().accent;
    if (kind == "fwd")                                           return Sty().info;
    if (kind == "train")                                         return Sty().good;
    return Sty().text_muted;
}

void DrawLogStream(AppState& s) {
    char flag[64];
    std::snprintf(flag, sizeof flag, "%zu entries · tail %s", s.logs.size(),
                  s.liveLogs ? "ON" : "OFF");
    DrawTitleBar("logstream", "█", flag, "logstream", [&] {
        if (ImGui::SmallButton(s.liveLogs ? "|| tail" : "> tail")) s.liveLogs = !s.liveLogs;
        ImGui::SameLine();
        static char search[64] = {};
        ImGui::SetNextItemWidth(120);
        ImGui::InputTextWithHint("##s", "search…", search, sizeof search);
    });
    if (!ImGui::BeginChild("##log_body", ImVec2(0, 0), ImGuiChildFlags_None,
                            ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGui::EndChild(); return;
    }
    for (const auto& l : s.logs) {
        const auto secs = (l.ts_ms / 1000) % 86400;
        const int hh = int(secs / 3600);
        const int mm = int((secs / 60) % 60);
        const int ss = int(secs % 60);
        const int cs = int((l.ts_ms % 1000) / 10);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::Text("%02d:%02d:%02d.%02d", hh, mm, ss, cs);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, SevColor(l.kind));
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
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 32 || s.liveLogs)
        ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
}

void DrawFilters() {
    DrawTitleBar("filters", "⚲", nullptr, "filters");
    if (!ImGui::BeginChild("##fl_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    static std::unordered_map<std::string, bool> sev = {
        {"trace", true}, {"debug", true}, {"info", true}, {"warn", true}, {"error", true}, {"fatal", false}
    };
    static std::unordered_map<std::string, bool> src = {
        {"fwd", true}, {"train", true}, {"probe", true}, {"ablate", true}, {"hook", true},
        {"cache", true}, {"mem", true}, {"export", true}, {"inject", true}, {"run", true}
    };
    if (auto sec = BeginSection("severity", true)) {
        for (const auto* name : { "trace","debug","info","warn","error","fatal" }) {
            ImGui::Checkbox(name, &sev[name]);
        }
        EndSection(sec);
    }
    if (auto sec = BeginSection("source", true)) {
        for (const auto* name : { "fwd","train","probe","ablate","hook","cache","mem","export","inject","run" }) {
            ImGui::Checkbox(name, &src[name]);
        }
        EndSection(sec);
    }
    ImGui::EndChild();
}

void DrawMetrics(const AppState& s) {
    DrawTitleBar("metrics", "∿", nullptr, "metrics-panel");
    if (!ImGui::BeginChild("##mt_body", ImVec2(0, 0))) { ImGui::EndChild(); return; }
    if (auto sec = BeginSection("Live counters", true)) {
        char rate[32]; std::snprintf(rate, sizeof rate, "%zu /min", s.logs.size() / 60 + 1);
        char ren[32];  std::snprintf(ren, sizeof ren, "%.1f ms / step", s.renderMs);
        KV({
            { "log rate",  rate, "accent" },
            { "warn rate", "4 /min", "warn" },
            { "err rate",  "0 /min", "good" },
            { "cuda mem",  "38.4 / 80 GB", "good" },
            { "cpu",       "11.2%",        "" },
            { "fwd time",  ren,            "" },
        });
        EndSection(sec);
    }
    ImGui::EndChild();
}

}  // namespace

void DrawLogsWorkspace(AppState& s, Model& /*m*/) {
    const float W = ImGui::GetContentRegionAvail().x, H = ImGui::GetContentRegionAvail().y;
    const float gap = 1.0f;
    const float right_w = 320.0f;
    const float left_w  = std::max(200.0f, W - right_w - gap);
    const float bot_h   = 200.0f;
    const float top_h   = H - bot_h - gap;

    ImGui::BeginChild("##log_left", { left_w, H }, ImGuiChildFlags_Borders);
    DrawLogStream(s); ImGui::EndChild(); ImGui::SameLine(0, gap);

    ImGui::BeginChild("##log_right", { right_w, H });
    ImGui::BeginChild("##log_filters", { right_w, top_h }, ImGuiChildFlags_Borders);
    DrawFilters(); ImGui::EndChild();
    ImGui::BeginChild("##log_metrics", { right_w, bot_h }, ImGuiChildFlags_Borders);
    DrawMetrics(s); ImGui::EndChild();
    ImGui::EndChild();
}

}  // namespace llob
