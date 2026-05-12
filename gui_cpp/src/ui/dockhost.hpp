#pragma once
#include <imgui.h>

namespace llob {

// ── Per-workspace docking ─────────────────────────────────────────────────
// Each workspace owns its own DockSpace.  All 9 are submitted every frame
// (one visible, the others with KeepAliveOnly so docked panels don't pop
// out when their workspace is hidden).  Layouts persist in imgui.ini.
//
// The kind string MUST equal `WsDef(ws).short_label` — the dock id is
// hashed from it so the same workspace gets a stable ImGuiID across runs.

// Stable dockspace id for a workspace.  Same kind → same ImGuiID forever.
ImGuiID WorkspaceDockId(const char* kind);

// Submit the workspace's dockspace.  Pass `is_active=false` for the eight
// hidden workspaces each frame so their panels don't undock.  Must be
// called inside a parent window (e.g. ##dockhost).
void SubmitWorkspaceDockSpace(const char* kind, bool is_active);

// True exactly once per `kind` until ResetWorkspaceLayout(kind) is called
// or the imgui.ini state is wiped.  Use at the top of each workspace's
// panel-submit function to know when to (re-)build the DockBuilder tree.
bool ShouldBuildWorkspaceLayout(const char* kind);

// Force the named workspace to rebuild its layout on the next frame.
// Wired to View ▸ Reset workspace layout.
void ResetWorkspaceLayout(const char* kind);

}  // namespace llob
