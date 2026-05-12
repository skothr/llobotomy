#include "ui/dialogs.hpp"

#include "logger.hpp"
#include "style.hpp"

#include <imgui.h>

#include <chrono>
#include <cstdio>

namespace llob {

namespace {

constexpr const char* kVersion = "0.5.0";

#if LLOB_USE_MOCK_DATA
constexpr const char* kBuildMode = "mock-data";
#else
constexpr const char* kBuildMode = "engine-backend";
#endif

// Toast-dismiss state — separate from the log ring so we don't mutate
// AppState here.  Tracks the timestamp of the last entry already
// surfaced (so a subsequent error replaces it) and an explicit-dismiss
// timestamp (a click hides any toast at-or-older than this).
struct ToastState {
    std::int64_t shown_ts_ms     = 0;
    std::int64_t dismissed_ts_ms = 0;
};
ToastState& Toast() {
    static ToastState t;
    return t;
}

}  // namespace

// ── About dialog ───────────────────────────────────────────────────────────
void DrawAboutDialog(const AppState& s, bool request_open) {
    if (request_open) ImGui::OpenPopup("about##modal");

    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(420, 0), ImGuiCond_Appearing);

    if (!ImGui::BeginPopupModal("about##modal", nullptr,
                                ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse)) {
        return;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        return;
    }

    ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
    ImGui::TextUnformatted("\xE2\x96\x91 llobotomy");                  // ░
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
    ImGui::Text("v%s · %s", kVersion, kBuildMode);
    ImGui::PopStyleColor();

    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    ImGui::TextUnformatted("native C++ / Dear ImGui port of the React");
    ImGui::TextUnformatted("LLM-interpretability bench (llobotomy.html).");

    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
    ImGui::TextUnformatted("LOG FILE");
    ImGui::PopStyleColor();
    ImGui::TextUnformatted(LoggerPath().c_str());

    ImGui::Spacing();

    if (ImGui::Button("copy diagnostics", ImVec2(160, 0))) {
        // Put boot info on the system clipboard for paste-in-bug-report
        // workflows.  Includes version, build mode, log path, model
        // topology, current workspace.
        char diag[512];
        std::snprintf(diag, sizeof diag,
                      "llobotomy v%s · build=%s\n"
                      "log: %s\n"
                      "model: %s (%dL x %dh x %dd_model)\n"
                      "workspace: %s\n"
                      "render: %.1fms / %.0ffps\n",
                      kVersion, kBuildMode, LoggerPath().c_str(),
                      s.hasModel() ? s.model.name.c_str() : "—",
                      s.model.nLayers, s.model.nHeads, s.model.dModel,
                      WsDef(s.activeWs).short_label,
                      double(s.renderMs), double(s.fps));
        ImGui::SetClipboardText(diag);
        LLOB_LOG_INFO("about", "diagnostics copied to clipboard");
    }
    ImGui::SameLine();
    if (ImGui::Button("close", ImVec2(120, 0))) ImGui::CloseCurrentPopup();

    ImGui::EndPopup();
}

// ── Error toast ────────────────────────────────────────────────────────────
void DrawErrorToast(const AppState& s) {
    constexpr std::int64_t kVisibleMs = 5000;

    // Find the most recent entry with severity >= Warn that's still within
    // the visibility window.  Snapshot the tail under the log mutex; the
    // recent slice is small so the lock is held briefly.
    const auto recent = s.snapshotLogs(20);
    std::int64_t latest_ms = 0;
    const LogEntry* latest = nullptr;
    for (auto it = recent.rbegin(); it != recent.rend(); ++it) {
        if (static_cast<int>(it->sev) < static_cast<int>(Severity::Warn)) continue;
        latest_ms = it->ts_ms;
        latest    = &*it;
        break;
    }
    if (!latest) return;

    const std::int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (now_ms - latest_ms > kVisibleMs)              return;
    if (latest_ms <= Toast().dismissed_ts_ms)         return;
    Toast().shown_ts_ms = latest_ms;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float pad   = 12.0f;
    const float w     = 380.0f;
    const ImVec2 pos { vp->WorkPos.x + vp->WorkSize.x - w - pad,
                       vp->WorkPos.y + vp->WorkSize.y - pad - 22.0f /* status bar */ - 60.0f };
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(ImVec2(w, 0));

    const ImU32 accent = (latest->sev == Severity::Warn) ? Sty().warn : Sty().bad;
    ImGui::PushStyleColor(ImGuiCol_WindowBg, Sty().bg_panel_alt);
    ImGui::PushStyleColor(ImGuiCol_Border,   accent);
    ImGui::PushStyleVar  (ImGuiStyleVar_WindowBorderSize, 2.0f);

    if (ImGui::Begin("##error_toast", nullptr,
                     ImGuiWindowFlags_NoTitleBar      | ImGuiWindowFlags_NoResize    |
                     ImGuiWindowFlags_NoMove          | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoFocusOnAppearing |
                     ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::PushStyleColor(ImGuiCol_Text, accent);
        ImGui::Text("[%s] %s", SeverityShort(latest->sev), latest->kind.c_str());
        ImGui::PopStyleColor();
        ImGui::TextWrapped("%s", latest->msg.c_str());

        // Click anywhere on the banner to dismiss
        if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            Toast().dismissed_ts_ms = latest_ms;
        }
    }
    ImGui::End();

    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
}

}  // namespace llob
