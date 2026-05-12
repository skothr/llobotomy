#include "ui/dockhost.hpp"

#include <imgui_internal.h>          // DockBuilder*, DockNode flags

#include <cstdio>
#include <string>
#include <unordered_set>

namespace llob {

namespace {

// Workspaces whose layout has been built at least once this process.
// Can't rely on DockBuilderGetNode(id) == nullptr because the per-frame
// KeepAliveOnly submissions for inactive workspaces eagerly create empty
// dock nodes, defeating the "first frame" detection that pattern would
// otherwise give us.
std::unordered_set<std::string> g_built;

// Workspaces that asked for a layout rebuild on the next frame.
// Cleared by ShouldBuildWorkspaceLayout the moment it returns true.
std::unordered_set<std::string> g_pending_resets;

}  // namespace

ImGuiID WorkspaceDockId(const char* kind) {
    // Use ImHashStr so the ID is independent of the current ID stack.
    // GetID() hashes against the active window's ID stack, which means the
    // same string would resolve to a different ImGuiID depending on whether
    // we're inside ##dockhost (Begin pushes its ID) or at top level — and
    // we call this from both contexts.
    char buf[32];
    std::snprintf(buf, sizeof buf, "dock_%s", kind);
    return ImHashStr(buf);
}

void SubmitWorkspaceDockSpace(const char* kind, bool is_active) {
    const ImGuiID id = WorkspaceDockId(kind);
    // AutoHideTabBar collapses dock tab bars when only one window is docked
    // in a node, so DrawTitleBar serves as the only chrome strip in the
    // common single-pane case.  Tabbed nodes (e.g. inspector + raw view)
    // still get the ImGui tab bar so the user can switch between them.
    ImGuiDockNodeFlags flags = ImGuiDockNodeFlags_AutoHideTabBar;
    if (!is_active) flags |= ImGuiDockNodeFlags_KeepAliveOnly;
    ImGui::DockSpace(id, ImVec2(0, 0), flags);
}

bool ShouldBuildWorkspaceLayout(const char* kind) {
    const std::string key(kind);
    auto reset_it = g_pending_resets.find(key);
    const bool reset_requested = (reset_it != g_pending_resets.end());
    if (reset_requested) {
        g_pending_resets.erase(reset_it);
        g_built.insert(key);
        return true;
    }
    if (g_built.contains(key)) return false;

    // First call this run — but a prior session's imgui.ini may already have
    // a saved layout for this workspace.  If the dock node has split
    // children or docked windows, that's the user's persisted layout —
    // don't clobber it.
    const ImGuiID id = WorkspaceDockId(kind);
    ImGuiDockNode* node = ImGui::DockBuilderGetNode(id);
    const bool already_populated =
        (node != nullptr) &&
        (node->ChildNodes[0] != nullptr || !node->Windows.empty());

    g_built.insert(key);
    return !already_populated;
}

void ResetWorkspaceLayout(const char* kind) {
    g_pending_resets.insert(kind);
}

}  // namespace llob
