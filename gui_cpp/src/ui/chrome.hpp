#pragma once
#include <imgui.h>

#include <functional>
#include <initializer_list>
#include <string_view>

namespace llob {

// ── Custom title bar inside a docked ImGui window ──────────────────────────
// Call DrawTitleBar at the very top of a window's body; it paints the
// (icon · title — flag · ##dockId) header strip over `titlebar_h` pixels in
// the bg_titlebar color, then advances the cursor below it. The host window
// itself should be created with NoTitleBar so this strip *is* the chrome.
//
// `controls` is a callback that paints right-aligned buttons (e.g. zoom +/−)
// inside the title strip; pass {} when nothing is needed.
void DrawTitleBar(const char* title, const char* icon, const char* flag,
                  const char* dockId, std::function<void()> controls = {});

// Section: collapsible grouped block inside a window's body. Returns true
// when expanded — caller submits children only on true.
//
//   if (auto s = BeginSection("Selected component", true /*accent*/, "L02")) {
//       // ... rows ...
//   }
//   EndSection(s);
//
// The Begin/End pair encodes the expansion state in the returned struct so
// callers can use scoped guards (defined in chrome_scoped.hpp later) without
// double-bookkeeping.
struct SectionScope {
    bool open;
    explicit operator bool() const { return open; }
};
SectionScope BeginSection(const char* title, bool accent = false, const char* badge = nullptr);
void         EndSection(SectionScope s);

// ── KV grid ────────────────────────────────────────────────────────────────
struct KVRow { const char* k; std::string_view v; const char* tone = ""; };
void KV(std::initializer_list<KVRow> rows, bool dense = false);

// ── Inline horizontal bar (ProgressBar w/ explicit color + tabular label) ─
void Bar(float value /* 0..1 */, float width, float height, ImU32 color,
         const char* label = nullptr);

// ── Pill ──────────────────────────────────────────────────────────────────
// Small inline status chip with colored border + tinted background.
// `tone` ∈ {"accent", "warn", "good", "bad", "dim"}.
void Pill(const char* text, const char* tone = "accent", bool solid = false);

// ── Workspace tab strip + project tab strip ───────────────────────────────
// The workspace tab strip is custom-drawn (it has accent fill + per-id
// glyph prefixes from the spec) instead of using ImGui::BeginTabBar.
struct WorkspaceTabsResult { int hovered = -1; bool changed = false; };
WorkspaceTabsResult DrawWorkspaceTabs(int& active /* in/out 0..N-1 */,
                                      const char* const* labels, int n,
                                      const char* right_text = nullptr);

// Sub-style helper for an outline-tree row (left-bordered, color-tinted,
// hover-aware). Returns true on click.
struct TreeRowFlags {
    int   indent_px   = 0;
    bool  active      = false;     // accent background + left border
    bool  strikethru  = false;
    ImU32 fg          = 0;         // 0 = default text
    bool  selectable  = true;
};
bool TreeRow(const char* id_str, const char* label, TreeRowFlags f);

// ── Right-click context handling on a custom-drawn item ───────────────────
// Helper: returns true if the most recent invisible-button-style item was
// right-clicked.
bool WasRightClicked();

}  // namespace llob
