// llobotomy — native C++/Dear ImGui port of the React mock at
// /tmp/claude/design/llobotomy/project/llobotomy.html.  See HANDOFF.md for
// the full source map / state model / interaction contract.
//
// This file owns: window/loop bring-up, top-level shell (menubar / project
// tabs / workspace tabs / status bar), keyboard shortcuts, and dispatch into
// each workspace.  Per-workspace layout + widgets live under workspaces/.

#include "appstate.hpp"
#include "logger.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "ui/dockhost.hpp"
#include "ui/settings.hpp"
#include "workspaces/workspaces.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

void glfw_error(int err, const char* desc) {
    LLOB_LOG_WARN("glfw", "error %d: %s", err, desc);
}

// Menu intents — DrawMenubar AND HandleShortcuts both populate this so the
// main loop has a single dispatch surface.  Keeping it in one struct lets
// future polish phases (file dialogs, About) add slots without re-threading
// the menubar/shortcut signatures.
struct MenuActions {
    bool open_settings = false;
    bool open_about    = false;
    bool open_ckpt     = false;
    bool save_probe    = false;
    bool export_state  = false;
    bool new_project   = false;     // Ctrl+T — emulate the + tab button
    bool close_project = false;     // Ctrl+W — close active project tab
    bool quit          = false;     // Ctrl+Q
};

MenuActions DrawMenubar(llob::AppState& s, llob::Model& m, MenuActions act) {
    using namespace llob;
    if (ImGui::BeginMainMenuBar()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
        ImGui::TextUnformatted("\xE2\x96\x91 llobotomy");      // ░
        ImGui::PopStyleColor();
        ImGui::Spacing();
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open checkpoint…", "Ctrl+O")) act.open_ckpt    = true;
            if (ImGui::MenuItem("Save probe set…",  "Ctrl+S")) act.save_probe   = true;
            if (ImGui::MenuItem("Export state…",    "Ctrl+E")) act.export_state = true;
            ImGui::Separator();
            if (ImGui::MenuItem("New project",   "Ctrl+T")) act.new_project   = true;
            if (ImGui::MenuItem("Close project", "Ctrl+W")) act.close_project = true;
            ImGui::Separator();
            if (ImGui::MenuItem("Quit", "Ctrl+Q")) act.quit = true;
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))   { ImGui::MenuItem("Undo", "Ctrl+Z"); ImGui::MenuItem("Redo"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("Model"))  { ImGui::MenuItem("Reload"); ImGui::MenuItem("Switch model…"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("Probe"))  { ImGui::MenuItem("Train new probe…"); ImGui::MenuItem("Library"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("Run"))    { ImGui::MenuItem("Run / Pause", "Space"); ImGui::MenuItem("Step"); ImGui::MenuItem("Reset"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Settings…", "Ctrl+,")) act.open_settings = true;
            ImGui::Separator();
            ImGui::MenuItem("Show raw pane", nullptr, &s.showRaw);
            ImGui::MenuItem("Animate live data", nullptr, &s.liveAnim);
            ImGui::Separator();
            if (ImGui::MenuItem("Reset workspace layout")) {
                llob::ResetWorkspaceLayout(WsDef(s.activeWs).short_label);
                LLOB_LOG_INFO("ws", "reset layout for %s", WsDef(s.activeWs).short_label);
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help"))   { if (ImGui::MenuItem("About")) act.open_about = true; ImGui::EndMenu(); }

        // [DATA HOOK] Model::getEngineMetrics() — device tag, dtype,
        // current frame time, vram usage.  Renders as "—" when unwired.
        const auto em = m.getEngineMetrics();
        char right[160];
        char dev_buf[32], dtype_buf[16], frame_buf[24], mem_buf[32];
        std::snprintf(dev_buf,   sizeof dev_buf,   "%s", em.device.empty() ? "—" : em.device.c_str());
        std::snprintf(dtype_buf, sizeof dtype_buf, "%s", em.dtype.empty()  ? "—" : em.dtype.c_str());
        if (std::isnan(em.fwd_time_ms)) std::snprintf(frame_buf, sizeof frame_buf, "frame —");
        else                              std::snprintf(frame_buf, sizeof frame_buf, "frame %.1fms", double(em.fwd_time_ms));
        if (std::isnan(em.cuda_mem_used_GB) || std::isnan(em.cuda_mem_total_GB))
            std::snprintf(mem_buf, sizeof mem_buf, "mem —");
        else
            std::snprintf(mem_buf, sizeof mem_buf, "mem %.1f/%.0fG",
                          double(em.cuda_mem_used_GB), double(em.cuda_mem_total_GB));
        std::snprintf(right, sizeof right, "%s · %s · %s · %s",
                      dev_buf, dtype_buf, frame_buf, mem_buf);
        const float w = ImGui::CalcTextSize(right).x;
        ImGui::SameLine(ImGui::GetWindowWidth() - w - 16.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted(right);
        ImGui::PopStyleColor();
        ImGui::EndMainMenuBar();
    }
    return act;
}

void DrawStatusBar(const llob::AppState& s) {
    using namespace llob;
    ImGuiViewport* vp = ImGui::GetMainViewport();
    const float h = 22.0f;
    ImGui::SetNextWindowPos({vp->WorkPos.x, vp->WorkPos.y + vp->WorkSize.y - h});
    ImGui::SetNextWindowSize({vp->WorkSize.x, h});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, Sty().accent_dim);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 2));
    ImGui::Begin("##statusbar", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoSavedSettings |
                 ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus);
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 255));
    auto seg_text = [](const char* t) {
        ImGui::TextUnformatted(t);
        ImGui::SameLine(0, 16.0f);
    };
    seg_text(s.running ? "> RUNNING" : s.hasModel() ? "* CONNECTED" : "* NO MODEL");
    char buf[128];
    std::snprintf(buf, sizeof buf, "project: %s",
                  s.activeProject.empty() ? "—" : s.activeProject.c_str()); seg_text(buf);
    std::snprintf(buf, sizeof buf, "workspace: %s", WsDef(s.activeWs).short_label); seg_text(buf);
    if (s.hasModel()) std::snprintf(buf, sizeof buf, "L%02d/%d", s.activeLayer, s.model.nLayers);
    else              std::snprintf(buf, sizeof buf, "L—/—");
    seg_text(buf);
    if (!s.sampleTokens.empty()) std::snprintf(buf, sizeof buf, "tok %02d", s.activeToken);
    else                          std::snprintf(buf, sizeof buf, "tok —");
    seg_text(buf);
    ImGui::PopStyleColor();
    {
        char abuf[64];
        std::snprintf(abuf, sizeof abuf, "ablated: %zuh + %zuc",
                      s.ablatedHeads.size(), s.ablatedComponents.size());
        const ImU32 col = (s.ablatedHeads.size() + s.ablatedComponents.size() > 0)
                          ? IM_COL32(255, 200, 200, 255) : IM_COL32(255, 255, 255, 255);
        ImGui::PushStyleColor(ImGuiCol_Text, col);
        ImGui::TextUnformatted(abuf);
        ImGui::PopStyleColor();
        ImGui::SameLine(0, 16.0f);
    }
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 255));
    std::snprintf(buf, sizeof buf, "probes: %zu", s.probedHeads.size() + s.probedComponents.size()); seg_text(buf);
    std::snprintf(buf, sizeof buf, "theme: %s",
                  s.theme == llob::Theme::Tracy ? "tracy"
                : s.theme == llob::Theme::Photoshop ? "photoshop" : "amber"); seg_text(buf);
    ImGui::PopStyleColor();

    // Spacer
    const char* tail_left  = "kb: ↑↓ layer · ←→ token · space run · 1-9 ws";
    const char* tail_right = "build 0.4.18-rc2";
    const float right_w = ImGui::CalcTextSize(tail_right).x +
                          ImGui::CalcTextSize(tail_left).x + 80.0f;
    ImGui::SameLine(ImGui::GetWindowWidth() - right_w);
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 220));
    ImGui::TextUnformatted(tail_left);
    ImGui::SameLine(0, 16);
    char render[32]; std::snprintf(render, sizeof render, "render: %.1f ms", s.renderMs);
    ImGui::TextUnformatted(render);
    ImGui::SameLine(0, 16);
    ImGui::TextUnformatted(tail_right);
    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void DrawProjectTabs(llob::AppState& s) {
    using namespace llob;
    if (ImGui::BeginTabBar("##projects", ImGuiTabBarFlags_Reorderable | ImGuiTabBarFlags_AutoSelectNewTabs)) {
        // PushID(p.id) on each iteration so two tabs that happen to share a
        // display name still get distinct ImGui IDs — the previous version
        // hashed solely on the label, which collided every time the + button
        // produced a second "untitled".
        std::size_t closed_idx = SIZE_MAX;
        // Only update activeProject when the user actually clicks a tab
        // (IsItemActivated fires once per click).  When SetSelected forces
        // a new tab into focus, ImGui may briefly report the OLD tab as
        // open this frame — without this gate, our previous code would
        // overwrite the just-set activeProject and snap back.
        for (std::size_t i = 0; i < s.projects.size(); ++i) {
            auto& p = s.projects[i];
            ImGui::PushID(p.id.c_str());
            ImGuiTabItemFlags flags = (p.id == s.activeProject) ? ImGuiTabItemFlags_SetSelected : 0;
            bool open = true;
            if (ImGui::BeginTabItem(p.name.c_str(), &open, flags)) {
                if (ImGui::IsItemActivated()) s.activeProject = p.id;
                ImGui::EndTabItem();
            }
            ImGui::PopID();
            // Honor the close-X — defer the actual erase until after the
            // BeginTabBar loop so we don't invalidate iterators/IDs mid-frame.
            if (!open && closed_idx == SIZE_MAX) closed_idx = i;
        }
        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
            // Monotonic suffix keeps id + display label unique across clicks
            // so the next "untitled" doesn't collide with the previous one.
            static int next_untitled = 1;
            char id_buf[32], name_buf[32];
            std::snprintf(id_buf,   sizeof id_buf,   "p_untitled_%d", next_untitled);
            std::snprintf(name_buf, sizeof name_buf, "untitled-%d",   next_untitled);
            ++next_untitled;
            s.projects.push_back({ id_buf, name_buf, ProjectTab::Dot::Dim, "" });
        }
        ImGui::EndTabBar();

        if (closed_idx != SIZE_MAX && closed_idx < s.projects.size()) {
            const std::string was_active = s.activeProject;
            const std::string closed_id  = s.projects[closed_idx].id;
            s.projects.erase(s.projects.begin() + static_cast<std::ptrdiff_t>(closed_idx));
            // If the user closed the active project, snap to whatever's left
            // (or clear it when the list empties).
            if (was_active == closed_id) {
                s.activeProject = s.projects.empty() ? "" : s.projects.front().id;
            }
        }
    }
}

void DrawWorkspaceTabRow(llob::AppState& s) {
    using namespace llob;
    const char* labels[kNumWorkspaces];
    for (int i = 0; i < kNumWorkspaces; ++i) labels[i] = WsDef(static_cast<Workspace>(i)).label;

    char right[160];
    if (s.hasModel())
        std::snprintf(right, sizeof right, "model: %s  |  %dL x %dh x %dd",
                      s.model.name.c_str(), s.model.nLayers, s.model.nHeads, s.model.dModel);
    else
        std::snprintf(right, sizeof right, "model: —");

    int active = static_cast<int>(s.activeWs);
    DrawWorkspaceTabs(active, labels, kNumWorkspaces, right);
    if (active != static_cast<int>(s.activeWs)) {
        s.activeWs = static_cast<Workspace>(active);
        LLOB_LOG_DEBUG("ws", "switched workspace -> %s", WsDef(s.activeWs).short_label);
    }
}

// Pump keyboard shortcuts.  Ctrl-chords work even when text input has focus
// (standard editor convention); bare keys (Up/Down/Space/1-9) are gated on
// !typing so they don't fight a focused input box.  Some shortcuts populate
// MenuActions for the main loop to dispatch (consistent with menu clicks);
// others (workspace cycle, layer/token nudge) mutate AppState directly
// because there's no menu equivalent.
void HandleShortcuts(llob::AppState& s, MenuActions& act) {
    using namespace llob;
    const ImGuiIO& io = ImGui::GetIO();
    const bool ctrl   = io.KeyCtrl;
    const bool shift  = io.KeyShift;
    const bool typing = io.WantTextInput;

    // Ctrl-chord shortcuts — fire even during text input
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_Comma))  act.open_settings = true;
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_O))      act.open_ckpt     = true;
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_S))      act.save_probe    = true;
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_E))      act.export_state  = true;
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_T))      act.new_project   = true;
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_W))      act.close_project = true;
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_Q))      act.quit          = true;

    // Cycle workspace via Ctrl+Tab / Ctrl+Shift+Tab (browser-style)
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_Tab)) {
        int next = static_cast<int>(s.activeWs) + (shift ? -1 : +1);
        if (next < 0) next = kNumWorkspaces - 1;
        if (next >= kNumWorkspaces) next = 0;
        s.activeWs = static_cast<Workspace>(next);
        LLOB_LOG_DEBUG("ws", "cycled (%s) -> %s",
                       shift ? "back" : "fwd", WsDef(s.activeWs).short_label);
    }
    // Direct workspace jump via Ctrl+1..9
    if (ctrl) {
        for (int i = 0; i < kNumWorkspaces && i < 9; ++i) {
            if (ImGui::IsKeyPressed(static_cast<ImGuiKey>(int(ImGuiKey_1) + i))) {
                s.activeWs = static_cast<Workspace>(i);
                LLOB_LOG_DEBUG("ws", "Ctrl+%d -> %s", i + 1, WsDef(s.activeWs).short_label);
            }
        }
    }

    // Bare-key shortcuts only when not typing
    if (typing) return;
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow))    s.setActiveLayer(s.activeLayer - 1);
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow))  s.setActiveLayer(s.activeLayer + 1);
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))  s.setActiveToken(s.activeToken - 1);
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) s.setActiveToken(s.activeToken + 1);
    if (ImGui::IsKeyPressed(ImGuiKey_Space))      s.running = !s.running;
    if (!ctrl) {
        for (int i = 0; i < kNumWorkspaces; ++i) {
            if (ImGui::IsKeyPressed(static_cast<ImGuiKey>(int(ImGuiKey_1) + i))) {
                s.activeWs = static_cast<Workspace>(i);
                LLOB_LOG_DEBUG("ws", "key %d -> %s", i + 1, WsDef(s.activeWs).short_label);
            }
        }
    }
}

// Build the active workspace's dock-tree layout if it hasn't been built yet
// (or was reset via View ▸ Reset workspace layout).  Must be called BEFORE
// any DockSpace() submission, per the canonical DockBuilder pattern.
void BuildActiveLayoutIfNeeded(const llob::AppState& s) {
    using namespace llob;
    const char* kind = WsDef(s.activeWs).short_label;
    if (!ShouldBuildWorkspaceLayout(kind)) return;
    const ImGuiID id = WorkspaceDockId(kind);
    switch (s.activeWs) {
        case Workspace::Arch:   BuildArchitectureLayout(id); break;
        case Workspace::Inf:    BuildInferenceLayout   (id); break;
        case Workspace::Attn:   BuildAttentionLayout   (id); break;
        case Workspace::Probes: BuildProbesLayout      (id); break;
        case Workspace::Train:  BuildTrainingLayout    (id); break;
        case Workspace::Ft:     BuildFineTuneLayout    (id); break;
        case Workspace::Data:   BuildDatasetsLayout    (id); break;
        case Workspace::Raw:    BuildRawTensorsLayout  (id); break;
        case Workspace::Logs:   BuildLogsLayout        (id); break;
        case Workspace::Count:  break;
    }
}

// Submit the active workspace's panels as top-level windows.  They auto-
// route into the workspace's DockSpace via the prior DockBuilderDockWindow
// assignment.  Inactive workspaces' panels aren't submitted; their
// dockspaces stay alive via KeepAliveOnly so the dock layout persists.
void DispatchWorkspacePanels(llob::AppState& s, llob::Model& m) {
    using namespace llob;
    switch (s.activeWs) {
        case Workspace::Arch:   SubmitArchitecturePanels(s, m); break;
        case Workspace::Inf:    SubmitInferencePanels   (s, m); break;
        case Workspace::Attn:   SubmitAttentionPanels   (s, m); break;
        case Workspace::Probes: SubmitProbesPanels      (s, m); break;
        case Workspace::Train:  SubmitTrainingPanels    (s, m); break;
        case Workspace::Ft:     SubmitFineTunePanels    (s, m); break;
        case Workspace::Data:   SubmitDatasetsPanels    (s, m); break;
        case Workspace::Raw:    SubmitRawTensorsPanels  (s, m); break;
        case Workspace::Logs:   SubmitLogsPanels        (s, m); break;
        case Workspace::Count:  break;
    }
}

}  // namespace

int main() {
    glfwSetErrorCallback(glfw_error);
    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

    GLFWwindow* win = glfwCreateWindow(1600, 1000, "llobotomy", nullptr, nullptr);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Fonts: Ubuntu Mono as the primary face (matches the mock's tracy /
    // photoshop / amber feel — sharp + technical), with Noto Sans Symbols2
    // merged in for the geometric / arrow / math glyphs Ubuntu Mono lacks
    // (◫ ▶ ▦ ◈ ∿ ◆ ≡ █ ↑↓←→ ⊕ ⊘ ⌬ ❄ ⚙ ⚲ etc).
    static const ImWchar kSymbolRanges[] = {
        0x0020, 0x00FF,   // Basic Latin + Latin-1 Supplement
        0x2010, 0x205F,   // General Punctuation
        0x2190, 0x21FF,   // Arrows
        0x2200, 0x22FF,   // Mathematical Operators
        0x2300, 0x23FF,   // Miscellaneous Technical
        0x2500, 0x257F,   // Box Drawing
        0x2580, 0x259F,   // Block Elements
        0x25A0, 0x25FF,   // Geometric Shapes
        0x2600, 0x26FF,   // Miscellaneous Symbols
        0x2700, 0x27BF,   // Dingbats
        0,
    };
    const float kFontSize = 14.0f;
    const char* kPrimary  = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf";
    // Symbol fallbacks, merged in priority order (earlier wins on overlap):
    // JetBrains Mono first — it's the closest tonal match to Ubuntu Mono and
    // covers most of the Unicode chars Ubuntu Mono lacks (arrows, geometric
    // shapes, math operators) without metric jitter.  DejaVu Sans Mono fills
    // any remaining gaps with stable monospace metrics; Noto Sans Symbols2
    // mops up the long-tail geometric / dingbat glyphs.
    const char* kFallbacks[] = {
        "/home/skothr/.fonts/jetbrains-mono/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
        nullptr,
    };
    // ImGui asserts (rather than returning nullptr) on a missing file, so
    // existence-check every path before handing it to AddFontFromFileTTF.
    if (std::filesystem::exists(kPrimary) &&
        io.Fonts->AddFontFromFileTTF(kPrimary, kFontSize, nullptr, kSymbolRanges)) {
        ImFontConfig merge{};
        merge.MergeMode = true;
        for (const char** fp = kFallbacks; *fp; ++fp) {
            if (std::filesystem::exists(*fp)) {
                io.Fonts->AddFontFromFileTTF(*fp, kFontSize, &merge, kSymbolRanges);
            }
        }
    } else {
        LLOB_LOG_WARN("font", "Ubuntu Mono not found at %s — using ImGui default", kPrimary);
    }

    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    llob::AppState s;
    s.seedSession();
    // Logger has to come up early — every subsequent push goes to both
    // the in-memory ring AND the on-disk log file at the path below.
    llob::LoggerInit(&s, "./llobotomy.log");
    LLOB_LOG_INFO("init", "llobotomy starting · log file: %s",
                  llob::LoggerPath().c_str());
    // SettingsLoad applies the persisted theme/density/accent itself
    // (calls ApplyStyle internally), so no explicit BuildStyle here.
    llob::SettingsLoad(s);
#if LLOB_USE_MOCK_DATA
    s.seedMockData();
#endif

    llob::MockModel mm;
    llob::Model&    model = mm;
    s.loadFromModel(model);  // populate AppState.model from a real backend

    auto last = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        s.tickLiveFeed();
        // [DATA HOOK] Model::drainEngineLogs — pull any log lines the
        // engine has produced since the last frame and fan them through
        // the standard Logger so they hit both sinks.
        for (const auto& e : model.drainEngineLogs()) {
            llob::LoggerPush(e.sev, e.kind, e.msg);
        }
        // Shortcuts populate the action struct; menu clicks add to it.
        MenuActions menu{};
        HandleShortcuts(s, menu);
        menu = DrawMenubar(s, model, menu);

        // Dispatch wired actions
        llob::DrawSettingsModal(s, menu.open_settings);
        if (menu.quit) glfwSetWindowShouldClose(win, GLFW_TRUE);
        if (menu.new_project) {
            // Same logic as the + button — monotonic suffix keeps id +
            // display name unique so duplicates don't collide on ImGui IDs.
            static int next_untitled_kb = 1;
            char id_buf[32], name_buf[32];
            std::snprintf(id_buf,   sizeof id_buf,   "p_kb_untitled_%d", next_untitled_kb);
            std::snprintf(name_buf, sizeof name_buf, "untitled-%d",      next_untitled_kb);
            ++next_untitled_kb;
            s.projects.push_back({ id_buf, name_buf, llob::ProjectTab::Dot::Dim, "" });
            s.activeProject = id_buf;
            LLOB_LOG_INFO("project", "new project %s", name_buf);
        }
        if (menu.close_project && !s.projects.empty()) {
            LLOB_LOG_DEBUG("project", "close requested; activeProject=%s",
                           s.activeProject.empty() ? "(none)" : s.activeProject.c_str());
            auto it = std::find_if(s.projects.begin(), s.projects.end(),
                                   [&](const llob::ProjectTab& p){ return p.id == s.activeProject; });
            if (it != s.projects.end()) {
                const std::string name = it->name;
                s.projects.erase(it);
                s.activeProject = s.projects.empty() ? "" : s.projects.front().id;
                LLOB_LOG_INFO("project", "closed %s", name.c_str());
            }
        }
        // Phase 3 will dispatch menu.open_ckpt / save_probe / export_state
        // and Phase 7 menu.open_about.
        (void)menu.open_about; (void)menu.open_ckpt;
        (void)menu.save_probe; (void)menu.export_state;

        // Project tabs strip (just under the menubar).
        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({vp->WorkPos.x, vp->WorkPos.y});
        ImGui::SetNextWindowSize({vp->WorkSize.x, 28.0f});
        ImGui::Begin("##projtabs", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBringToFrontOnFocus);
        DrawProjectTabs(s);
        ImGui::End();

        // Workspace tab strip.
        ImGui::SetNextWindowPos({vp->WorkPos.x, vp->WorkPos.y + 28.0f});
        ImGui::SetNextWindowSize({vp->WorkSize.x, 26.0f});
        ImGui::Begin("##wstabs", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBringToFrontOnFocus);
        DrawWorkspaceTabRow(s);
        ImGui::End();

        // Workspace dockspace host.  Build the active workspace's layout
        // BEFORE submitting any DockSpace (per the canonical DockBuilder
        // pattern); inactive workspaces' dockspaces are still submitted
        // each frame with KeepAliveOnly so their docked panels don't pop
        // out when the user tabs away.
        BuildActiveLayoutIfNeeded(s);

        const float top    = 28.0f + 26.0f;
        const float bottom = 22.0f;
        ImGui::SetNextWindowPos({vp->WorkPos.x, vp->WorkPos.y + top});
        ImGui::SetNextWindowSize({vp->WorkSize.x, vp->WorkSize.y - top - bottom});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("##dockhost", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus);
        ImGui::PopStyleVar();
        const char* active_kind = llob::WsDef(s.activeWs).short_label;
        llob::SubmitWorkspaceDockSpace(active_kind, /*is_active=*/true);
        for (int i = 0; i < llob::kNumWorkspaces; ++i) {
            if (i == static_cast<int>(s.activeWs)) continue;
            llob::SubmitWorkspaceDockSpace(
                llob::WsDef(static_cast<llob::Workspace>(i)).short_label,
                /*is_active=*/false);
        }
        ImGui::End();

        // Panels live at top level — they auto-route into the workspace's
        // dockspace via the prior DockBuilderDockWindow assignment.
        DispatchWorkspacePanels(s, model);

        DrawStatusBar(s);

        // Frame timing
        const auto now = std::chrono::steady_clock::now();
        const float ms = std::chrono::duration<float, std::milli>(now - last).count();
        last = now;
        s.renderMs = 0.9f * s.renderMs + 0.1f * ms;
        if (ms > 0.0f) s.fps = 0.9f * s.fps + 0.1f * (1000.0f / ms);

        ImGui::Render();
        int w, h; glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.054f, 0.066f, 0.086f, 1.0f);     // bg-deep
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    LLOB_LOG_INFO("init", "shutting down");
    llob::LoggerShutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
