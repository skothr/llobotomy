// llobotomy — native C++/Dear ImGui port of the React mock at
// /tmp/claude/design/llobotomy/project/llobotomy.html.  See HANDOFF.md for
// the full source map / state model / interaction contract.
//
// This file owns: window/loop bring-up, top-level shell (menubar / project
// tabs / workspace tabs / status bar), keyboard shortcuts, and dispatch into
// each workspace.  Per-workspace layout + widgets live under workspaces/.

#include "appstate.hpp"
#include "model/model.hpp"
#include "style.hpp"
#include "ui/chrome.hpp"
#include "workspaces/workspaces.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace {

void glfw_error(int err, const char* desc) {
    std::fprintf(stderr, "[glfw] error %d: %s\n", err, desc);
}

void DrawMenubar(const llob::AppState& s) {
    using namespace llob;
    if (ImGui::BeginMainMenuBar()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().accent);
        ImGui::TextUnformatted("\xE2\x96\x91 llobotomy");      // ░
        ImGui::PopStyleColor();
        ImGui::Spacing();
        if (ImGui::BeginMenu("File")) {
            ImGui::MenuItem("Open checkpoint…",     "Ctrl+O");
            ImGui::MenuItem("Save probe set…",      "Ctrl+S");
            ImGui::MenuItem("Export state…",        "Ctrl+E");
            ImGui::Separator();
            if (ImGui::MenuItem("Quit", "Ctrl+Q")) std::exit(0);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))   { ImGui::MenuItem("Undo", "Ctrl+Z"); ImGui::MenuItem("Redo"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("Model"))  { ImGui::MenuItem("Reload"); ImGui::MenuItem("Switch model…"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("Probe"))  { ImGui::MenuItem("Train new probe…"); ImGui::MenuItem("Library"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("Run"))    { ImGui::MenuItem("Run / Pause", "Space"); ImGui::MenuItem("Step"); ImGui::MenuItem("Reset"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("View"))   { ImGui::MenuItem("Settings…"); ImGui::Separator(); ImGui::MenuItem("Show raw pane"); ImGui::EndMenu(); }
        if (ImGui::BeginMenu("Help"))   { ImGui::MenuItem("About"); ImGui::EndMenu(); }

        // Right-aligned status block.
        const char* right = "cuda:0 A100 · fp16 · 14.2ms · 38.4/80G";
        const float w = ImGui::CalcTextSize(right).x;
        ImGui::SameLine(ImGui::GetWindowWidth() - w - 16.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
        ImGui::TextUnformatted(right);
        ImGui::PopStyleColor();
        (void)s;
        ImGui::EndMainMenuBar();
    }
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
    seg_text(s.running ? "> RUNNING" : "* CONNECTED");
    char buf[128];
    std::snprintf(buf, sizeof buf, "project: %s", s.activeProject.c_str()); seg_text(buf);
    std::snprintf(buf, sizeof buf, "workspace: %s", WsDef(s.activeWs).short_label); seg_text(buf);
    std::snprintf(buf, sizeof buf, "L%02d/%d", s.activeLayer, s.model.nLayers); seg_text(buf);
    std::snprintf(buf, sizeof buf, "tok %02d", s.activeToken); seg_text(buf);
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
        for (auto& p : s.projects) {
            ImGuiTabItemFlags flags = (p.id == s.activeProject) ? ImGuiTabItemFlags_SetSelected : 0;
            bool open = true;
            if (ImGui::BeginTabItem(p.name.c_str(), &open, flags)) {
                s.activeProject = p.id;
                ImGui::EndTabItem();
            }
            (void)open;  // close icon present, but don't actually mutate vector while iterating
        }
        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
            s.projects.push_back({"px", "untitled", ProjectTab::Dot::Dim, ""});
        }
        ImGui::EndTabBar();
    }
}

void DrawWorkspaceTabRow(llob::AppState& s) {
    using namespace llob;
    const char* labels[kNumWorkspaces];
    for (int i = 0; i < kNumWorkspaces; ++i) labels[i] = WsDef(static_cast<Workspace>(i)).label;

    char right[160];
    std::snprintf(right, sizeof right, "model: %s  |  %dL x %dh x %dd",
                  s.model.name.c_str(), s.model.nLayers, s.model.nHeads, s.model.dModel);

    int active = static_cast<int>(s.activeWs);
    DrawWorkspaceTabs(active, labels, kNumWorkspaces, right);
    if (active != static_cast<int>(s.activeWs)) {
        s.activeWs = static_cast<Workspace>(active);
        s.pushLog("init", std::string("switched workspace -> ") + WsDef(s.activeWs).short_label);
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
        std::fprintf(stderr, "[font] Ubuntu Mono not found at %s — using ImGui default\n",
                     kPrimary);
    }

    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    llob::AppState s; s.seedDemo();
    llob::ApplyStyle(llob::BuildStyle(s.theme, s.density, s.accentOverride));

    llob::MockModel mm;
    llob::Model&    model = mm;

    auto last = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        s.tickLiveFeed();
        HandleShortcuts(s);
        DrawMenubar(s);

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

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
