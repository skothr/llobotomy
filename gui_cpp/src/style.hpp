#pragma once
#include <imgui.h>

namespace llob {

enum class Theme : int { Tracy = 0, Photoshop = 1, Amber = 2 };
enum class Density : int { Compact = 0, Normal = 1, Spacious = 2 };

// All colors stored as packed ImU32 (AABBGGRR — IM_COL32 layout) so DrawList
// calls can use them directly without conversion. Float ImVec4 forms are
// generated on demand from a single source of truth.
struct Style {
    // Background ramp
    ImU32 bg_deep;
    ImU32 bg_window;
    ImU32 bg_panel;
    ImU32 bg_panel_alt;
    ImU32 bg_header;
    ImU32 bg_header_active;
    ImU32 bg_input;
    ImU32 bg_input_hover;
    ImU32 bg_tab;
    ImU32 bg_tab_active;
    ImU32 bg_tab_hover;
    ImU32 bg_titlebar;

    // Borders
    ImU32 border;
    ImU32 border_strong;
    ImU32 border_focus;
    ImU32 separator;

    // Text
    ImU32 text;
    ImU32 text_muted;
    ImU32 text_dim;
    ImU32 text_bright;
    ImU32 text_disabled;

    // Accent (theme-tunable)
    ImU32 accent;
    ImU32 accent_dim;
    ImU32 accent_bg;
    ImU32 accent_bg_strong;

    // Semantic
    ImU32 warn;
    ImU32 warn_dim;
    ImU32 warn_bg;
    ImU32 good;
    ImU32 good_bg;
    ImU32 bad;
    ImU32 bad_bg;
    ImU32 info;
    ImU32 magenta;
    ImU32 yellow;

    // Heatmap stops (low → high)
    ImU32 heat[5];

    // Density-derived metrics (px)
    float row_h;
    float fs_xs, fs_sm, fs_md, fs_lg, fs_xl;
};

// Build a Style from the theme tokens defined in HANDOFF.md §8 / theme.css.
// `accent` is a user-tunable RGBA override applied on top of the base palette.
Style BuildStyle(Theme t, Density d, ImU32 accent_override = 0);

// Push the Style's palette into the active ImGuiStyle (Colors[]) and apply
// density spacing. Idempotent — call whenever theme/density/accent changes.
void  ApplyStyle(const Style& s);

// Convenience colormaps (port of imgui-core.jsx heatColor + divergeColor).
ImU32 HeatColor(float v);              // v in [0,1] → blue→cyan→amber→red
ImU32 DivergeColor(float v);           // v in [-1,1] → blue→bg→orange
ImU32 LerpColor(ImU32 a, ImU32 b, float t);
ImU32 WithAlpha(ImU32 c, float a);     // a in [0,1]

// Return current style — set by main() once per frame (after Apply).
const Style& Sty();

}  // namespace llob
