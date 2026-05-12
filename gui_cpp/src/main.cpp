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
#include "ui/settings.hpp"
#include "workspaces/workspaces.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

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

// Menu intents — DrawMenubar collects the user's clicks here so the main
// loop can dispatch into the matching modal/dialog/action.  Keeping this in
// one struct lets future polish phases (file dialogs, About) add slots
// without re-threading the menubar signature.
struct MenuActions {
    bool open_settings = false;
    bool open_about    = false;
    bool open_ckpt     = false;
    bool save_probe    = false;
    bool export_state  = false;
};

MenuActions DrawMenubar(llob::AppState& s, llob::Model& m) {
    using namespace llob;
    MenuActions act{};
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
            if (ImGui::MenuItem("Quit", "Ctrl+Q")) std::exit(0);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))   { ImGui::MenuItem("Undo", "Ctrl+Z"); ImGui::MenuItem("Redo"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("Model"))  { ImGui::MenuItem("Reload"); ImGui::MenuItem("Switch model…"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("Probe"))  { ImGui::MenuItem("Train new probe…"); ImGui::MenuItem("Library"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("Run"))    { ImGui::MenuItem("Run / Pause", "Space"); ImGui::MenuItem("Step"); ImGui::MenuItem("Reset"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Settings…")) act.open_settings = true;
            ImGui::Separator();
            ImGui::MenuItem("Show raw pane", nullptr, &s.showRaw);
            ImGui::MenuItem("Animate live data", nullptr, &s.liveAnim);
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
        for (std::size_t i = 0; i < s.projects.size(); ++i) {
            auto& p = s.projects[i];
            ImGui::PushID(p.id.c_str());
            ImGuiTabItemFlags flags = (p.id == s.activeProject) ? ImGuiTabItemFlags_SetSelected : 0;
            bool open = true;
            if (ImGui::BeginTabItem(p.name.c_str(), &open, flags)) {
                s.activeProject = p.id;
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

void HandleShortcuts(llob::AppState& s) {
    using namespace llob;
    if (ImGui::GetIO().WantTextInput) return;
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow))    s.setActiveLayer(s.activeLayer - 1);
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow))  s.setActiveLayer(s.activeLayer + 1);
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))  s.setActiveToken(s.activeToken - 1);
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) s.setActiveToken(s.activeToken + 1);
    if (ImGui::IsKeyPressed(ImGuiKey_Space))      s.running = !s.running;
    for (int i = 0; i < kNumWorkspaces; ++i) {
        if (ImGui::IsKeyPressed(static_cast<ImGuiKey>(int(ImGuiKey_1) + i))) {
            s.activeWs = static_cast<Workspace>(i);
        }
    }
}

void DispatchWorkspace(llob::AppState& s, llob::Model& m) {
    using namespace llob;
    switch (s.activeWs) {
        case Workspace::Arch:   DrawArchitectureWorkspace(s, m); break;
        case Workspace::Inf:    DrawInferenceWorkspace   (s, m); break;
        case Workspace::Attn:   DrawAttentionWorkspace   (s, m); break;
        case Workspace::Probes: DrawProbesWorkspace      (s, m); break;
        case Workspace::Train:  DrawTrainingWorkspace    (s, m); break;
        case Workspace::Ft:     DrawFineTuneWorkspace    (s, m); break;
        case Workspace::Data:   DrawDatasetsWorkspace    (s, m); break;
        case Workspace::Raw:    DrawRawTensorsWorkspace  (s, m); break;
        case Workspace::Logs:   DrawLogsWorkspace        (s, m); break;
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
        HandleShortcuts(s);
        const MenuActions menu = DrawMenubar(s, model);
        llob::DrawSettingsModal(s, menu.open_settings);
        // Phase 3 will dispatch menu.open_ckpt / save_probe / export_state
        // and Phase 7 menu.open_about.  No-op for now — the bools are
        // captured so the menu items don't silently swallow the click.
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

        // Workspace dockspace host.
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
        DispatchWorkspace(s, model);
        ImGui::End();

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
