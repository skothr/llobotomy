#include "style.hpp"

#include <algorithm>
#include <cstdint>

namespace llob {

namespace {

constexpr ImU32 RGB(int r, int g, int b, int a = 255) {
    return IM_COL32(r, g, b, a);
}
constexpr ImU32 RGB(std::uint32_t hex, int a = 255) {
    return IM_COL32((hex >> 16) & 0xff, (hex >> 8) & 0xff, hex & 0xff, a);
}

Style g_current{};

}  // namespace

const Style& Sty() { return g_current; }

ImU32 LerpColor(ImU32 a, ImU32 b, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    auto chan = [&](int shift) {
        const int ca = (a >> shift) & 0xff;
        const int cb = (b >> shift) & 0xff;
        return static_cast<int>(ca + (cb - ca) * t) & 0xff;
    };
    return ImU32(chan(0)) | (ImU32(chan(8)) << 8) | (ImU32(chan(16)) << 16) | (ImU32(chan(24)) << 24);
}

ImU32 WithAlpha(ImU32 c, float a) {
    a = std::clamp(a, 0.0f, 1.0f);
    return (c & 0x00ffffff) | (static_cast<ImU32>(a * 255) << 24);
}

// Mirrors imgui-core.jsx::heatColor — five-stop ramp.
ImU32 HeatColor(float v) {
    v = std::clamp(v, 0.0f, 1.0f);
    const ImU32 stops[5] = {
        RGB(26, 36, 56),    // 0.00 deep blue
        RGB(45, 74, 107),   // 0.25
        RGB(61, 139, 148),  // 0.50 teal
        RGB(214, 160, 64),  // 0.75 amber
        RGB(224, 82, 74),   // 1.00 red
    };
    const float scaled = v * 4.0f;
    const int   idx    = std::min(3, static_cast<int>(scaled));
    return LerpColor(stops[idx], stops[idx + 1], scaled - float(idx));
}

// Mirrors imgui-core.jsx::divergeColor — bipolar [-1,1].
ImU32 DivergeColor(float v) {
    v = std::clamp(v, -1.0f, 1.0f);
    if (v >= 0.0f) {
        // bg → orange
        return LerpColor(RGB(26, 31, 39), RGB(232, 144, 66), v);
    }
    return LerpColor(RGB(26, 31, 39), RGB(78, 170, 210), -v);
}

Style BuildStyle(Theme t, Density d, ImU32 accent_override) {
    Style s{};

    // ── Tracy (default) palette ────────────────────────────────────────────
    s.bg_deep         = RGB(0x0e1116u);
    s.bg_window       = RGB(0x15191fu);
    s.bg_panel        = RGB(0x1a1f27u);
    s.bg_panel_alt    = RGB(0x1d232cu);
    s.bg_header       = RGB(0x232a35u);
    s.bg_header_active= RGB(0x2d3744u);
    s.bg_input        = RGB(0x0c0f14u);
    s.bg_input_hover  = RGB(0x131820u);
    s.bg_tab          = RGB(0x1a1f27u);
    s.bg_tab_active   = RGB(0x2a3442u);
    s.bg_tab_hover    = RGB(0x232a35u);
    s.bg_titlebar     = RGB(0x0a0d12u);

    s.border          = RGB(0x2a323du);
    s.border_strong   = RGB(0x384352u);
    s.border_focus    = RGB(0x4a5a70u);
    s.separator       = RGB(0x232a35u);

    s.text            = RGB(0xd9dee5u);
    s.text_muted      = RGB(0x8a95a3u);
    s.text_dim        = RGB(0x5e6773u);
    s.text_bright     = RGB(0xf1f4f8u);
    s.text_disabled   = RGB(0x4a525eu);

    s.accent          = RGB(0x4ec9d4u);
    s.accent_dim      = RGB(0x2d8088u);
    s.accent_bg       = WithAlpha(s.accent, 0.12f);
    s.accent_bg_strong= WithAlpha(s.accent, 0.22f);

    s.warn            = RGB(0xe89042u);
    s.warn_dim        = RGB(0xa86023u);
    s.warn_bg         = WithAlpha(s.warn, 0.14f);
    s.good            = RGB(0x6cc070u);
    s.good_bg         = WithAlpha(s.good, 0.14f);
    s.bad             = RGB(0xe0524au);
    s.bad_bg          = WithAlpha(s.bad, 0.14f);
    s.info            = RGB(0x8ab4ffu);
    s.magenta         = RGB(0xc878d6u);
    s.yellow          = RGB(0xd8c84au);

    s.heat[0] = RGB(0x1a2438u);
    s.heat[1] = RGB(0x2d4a6bu);
    s.heat[2] = RGB(0x3d8b94u);
    s.heat[3] = RGB(0xd6a040u);
    s.heat[4] = RGB(0xe0524au);

    // ── Theme overrides ────────────────────────────────────────────────────
    if (t == Theme::Photoshop) {
        s.bg_deep         = RGB(0x1c1c1cu);
        s.bg_window       = RGB(0x232323u);
        s.bg_panel        = RGB(0x2b2b2bu);
        s.bg_panel_alt    = RGB(0x303030u);
        s.bg_header       = RGB(0x383838u);
        s.bg_header_active= RGB(0x444444u);
        s.bg_input        = RGB(0x1a1a1au);
        s.bg_input_hover  = RGB(0x232323u);
        s.bg_tab          = RGB(0x2b2b2bu);
        s.bg_tab_active   = RGB(0x444444u);
        s.bg_tab_hover    = RGB(0x383838u);
        s.bg_titlebar     = RGB(0x161616u);
        s.border          = RGB(0x3d3d3du);
        s.border_strong   = RGB(0x4d4d4du);
        s.separator       = RGB(0x353535u);
        s.accent          = RGB(0x2d8fd8u);
        s.accent_bg       = WithAlpha(s.accent, 0.14f);
        s.accent_bg_strong= WithAlpha(s.accent, 0.24f);
    } else if (t == Theme::Amber) {
        s.accent          = RGB(0xffb85au);
        s.accent_dim      = RGB(0xa06a1fu);
        s.accent_bg       = WithAlpha(s.accent, 0.12f);
        s.accent_bg_strong= WithAlpha(s.accent, 0.22f);
        s.warn            = RGB(0x4ec9d4u);
        s.info            = RGB(0x4ec9d4u);
    }

    // User-tunable accent override (applied on top of theme).
    if (accent_override != 0) {
        s.accent          = accent_override | 0xff000000u;
        s.accent_bg       = WithAlpha(s.accent, 0.12f);
        s.accent_bg_strong= WithAlpha(s.accent, 0.22f);
    }

    // ── Density: spacing + font sizes ─────────────────────────────────────
    switch (d) {
        case Density::Compact:
            s.fs_xs = 9;  s.fs_sm = 10; s.fs_md = 11; s.fs_lg = 12; s.fs_xl = 14; s.row_h = 18;
            break;
        case Density::Spacious:
            s.fs_xs = 11; s.fs_sm = 12; s.fs_md = 13; s.fs_lg = 14; s.fs_xl = 16; s.row_h = 26;
            break;
        case Density::Normal:
        default:
            s.fs_xs = 10; s.fs_sm = 11; s.fs_md = 12; s.fs_lg = 13; s.fs_xl = 15; s.row_h = 22;
            break;
    }
    return s;
}

void ApplyStyle(const Style& s) {
    g_current = s;

    ImGuiStyle& gs = ImGui::GetStyle();
    auto V = [](ImU32 c) { return ImGui::ColorConvertU32ToFloat4(c); };

    // Sharp corners — tracy/photoshop look.
    gs.WindowRounding   = 0.f;
    gs.FrameRounding    = 0.f;
    gs.GrabRounding     = 0.f;
    gs.TabRounding      = 0.f;
    gs.PopupRounding    = 0.f;
    gs.ScrollbarRounding= 0.f;
    gs.ChildRounding    = 0.f;

    gs.WindowBorderSize = 1.f;
    gs.FrameBorderSize  = 1.f;
    gs.ChildBorderSize  = 1.f;
    gs.PopupBorderSize  = 1.f;

    // Density-dependent spacing.
    const float dy = (s.row_h - 22.f) * 0.5f;
    gs.WindowPadding   = ImVec2(6, 6);
    gs.FramePadding    = ImVec2(6, 3 + dy);
    gs.ItemSpacing     = ImVec2(6, 4 + dy);
    gs.ItemInnerSpacing= ImVec2(4, 4);
    gs.CellPadding     = ImVec2(6, 2 + dy);
    gs.IndentSpacing   = 14.f;

    // Color slot mapping (HANDOFF.md §8 table).
    gs.Colors[ImGuiCol_WindowBg]            = V(s.bg_window);
    gs.Colors[ImGuiCol_ChildBg]             = V(s.bg_window);
    gs.Colors[ImGuiCol_PopupBg]             = V(s.bg_panel);
    gs.Colors[ImGuiCol_Border]              = V(s.border);
    gs.Colors[ImGuiCol_BorderShadow]        = ImVec4(0, 0, 0, 0);
    gs.Colors[ImGuiCol_FrameBg]             = V(s.bg_input);
    gs.Colors[ImGuiCol_FrameBgHovered]      = V(s.bg_input_hover);
    gs.Colors[ImGuiCol_FrameBgActive]       = V(s.bg_input_hover);
    gs.Colors[ImGuiCol_TitleBg]             = V(s.bg_titlebar);
    gs.Colors[ImGuiCol_TitleBgActive]       = V(s.bg_titlebar);
    gs.Colors[ImGuiCol_TitleBgCollapsed]    = V(s.bg_titlebar);
    gs.Colors[ImGuiCol_MenuBarBg]           = V(s.bg_titlebar);
    gs.Colors[ImGuiCol_ScrollbarBg]         = V(s.bg_deep);
    gs.Colors[ImGuiCol_ScrollbarGrab]       = V(s.bg_header);
    gs.Colors[ImGuiCol_ScrollbarGrabHovered]= V(s.bg_header_active);
    gs.Colors[ImGuiCol_ScrollbarGrabActive] = V(s.border_focus);
    gs.Colors[ImGuiCol_CheckMark]           = V(s.accent);
    gs.Colors[ImGuiCol_SliderGrab]          = V(s.accent);
    gs.Colors[ImGuiCol_SliderGrabActive]    = V(s.accent);
    gs.Colors[ImGuiCol_Button]              = V(s.bg_header);
    gs.Colors[ImGuiCol_ButtonHovered]       = V(s.bg_header_active);
    gs.Colors[ImGuiCol_ButtonActive]        = V(s.bg_input);
    gs.Colors[ImGuiCol_Header]              = V(s.bg_header);
    gs.Colors[ImGuiCol_HeaderHovered]       = V(s.bg_header_active);
    gs.Colors[ImGuiCol_HeaderActive]        = V(s.accent_bg);
    gs.Colors[ImGuiCol_Separator]           = V(s.separator);
    gs.Colors[ImGuiCol_SeparatorHovered]    = V(s.border_focus);
    gs.Colors[ImGuiCol_SeparatorActive]     = V(s.accent);
    gs.Colors[ImGuiCol_ResizeGrip]          = V(s.border);
    gs.Colors[ImGuiCol_ResizeGripHovered]   = V(s.border_focus);
    gs.Colors[ImGuiCol_ResizeGripActive]    = V(s.accent);
    gs.Colors[ImGuiCol_Tab]                 = V(s.bg_tab);
    gs.Colors[ImGuiCol_TabHovered]          = V(s.bg_tab_hover);
    gs.Colors[ImGuiCol_TabSelected]         = V(s.bg_tab_active);
    gs.Colors[ImGuiCol_TabSelectedOverline] = V(s.accent);
    gs.Colors[ImGuiCol_TabDimmed]           = V(s.bg_tab);
    gs.Colors[ImGuiCol_TabDimmedSelected]   = V(s.bg_tab_active);
    gs.Colors[ImGuiCol_DockingPreview]      = V(s.accent_bg_strong);
    gs.Colors[ImGuiCol_DockingEmptyBg]      = V(s.bg_deep);
    gs.Colors[ImGuiCol_PlotLines]           = V(s.accent);
    gs.Colors[ImGuiCol_PlotLinesHovered]    = V(s.accent);
    gs.Colors[ImGuiCol_PlotHistogram]       = V(s.warn);
    gs.Colors[ImGuiCol_PlotHistogramHovered]= V(s.warn);
    gs.Colors[ImGuiCol_TableHeaderBg]       = V(s.bg_header);
    gs.Colors[ImGuiCol_TableBorderStrong]   = V(s.border_strong);
    gs.Colors[ImGuiCol_TableBorderLight]    = V(s.separator);
    gs.Colors[ImGuiCol_TableRowBg]          = V(s.bg_window);
    gs.Colors[ImGuiCol_TableRowBgAlt]       = V(s.bg_panel_alt);
    gs.Colors[ImGuiCol_TextSelectedBg]      = V(s.accent_bg_strong);
    gs.Colors[ImGuiCol_DragDropTarget]      = V(s.accent);
    gs.Colors[ImGuiCol_NavCursor]           = V(s.accent);
    gs.Colors[ImGuiCol_Text]                = V(s.text);
    gs.Colors[ImGuiCol_TextDisabled]        = V(s.text_disabled);
    gs.Colors[ImGuiCol_NavWindowingHighlight]= V(s.accent);
    gs.Colors[ImGuiCol_NavWindowingDimBg]   = ImVec4(0, 0, 0, 0.4f);
    gs.Colors[ImGuiCol_ModalWindowDimBg]    = ImVec4(0, 0, 0, 0.4f);
}

}  // namespace llob
