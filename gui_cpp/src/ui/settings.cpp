// Settings dialog + persisted prefs for theme / density / accent / showRaw /
// liveAnim.  Hand-rolled minimal JSON read/write — no third-party dep — kept
// honest by writing only a fixed object schema and parsing the same fields
// back; anything outside the schema is ignored.

#include "ui/settings.hpp"

#include "logger.hpp"
#include "style.hpp"

#include <imgui.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

namespace llob {

namespace {

const char* ThemeName(Theme t) {
    switch (t) {
        case Theme::Tracy:     return "tracy";
        case Theme::Photoshop: return "photoshop";
        case Theme::Amber:     return "amber";
    }
    return "tracy";
}

bool ThemeFromName(std::string_view s, Theme& out) {
    if (s == "tracy")     { out = Theme::Tracy;     return true; }
    if (s == "photoshop") { out = Theme::Photoshop; return true; }
    if (s == "amber")     { out = Theme::Amber;     return true; }
    return false;
}

const char* DensityName(Density d) {
    switch (d) {
        case Density::Compact:  return "compact";
        case Density::Normal:   return "normal";
        case Density::Spacious: return "spacious";
    }
    return "normal";
}

bool DensityFromName(std::string_view s, Density& out) {
    if (s == "compact")  { out = Density::Compact;  return true; }
    if (s == "normal")   { out = Density::Normal;   return true; }
    if (s == "spacious") { out = Density::Spacious; return true; }
    return false;
}

// Strip surrounding whitespace + optional quotes from a string view.
std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\n' || s.front() == '\r')) s.remove_prefix(1);
    while (!s.empty() && (s.back()  == ' ' || s.back()  == '\t' || s.back()  == '\n' || s.back()  == '\r' || s.back() == ',')) s.remove_suffix(1);
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        s.remove_prefix(1); s.remove_suffix(1);
    }
    return s;
}

// Apply theme/density/accent to the active ImGuiStyle.  Called after every
// modal change AND after SettingsLoad so the persisted prefs take effect.
void ReapplyStyle(const AppState& s) {
    ApplyStyle(BuildStyle(s.theme, s.density, s.accentOverride));
}

void EnsureParentDir(const std::filesystem::path& p) {
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    if (ec) {
        LLOB_LOG_WARN("settings", "could not create %s: %s",
                      p.parent_path().c_str(), ec.message().c_str());
    }
}

// Five preset accent swatches — keep them visually distinct from the three
// theme defaults so the override mode is visible at a glance.
struct AccentPreset { const char* name; ImU32 color; };
constexpr AccentPreset kAccentPresets[] = {
    { "cyan",    IM_COL32(0x4e, 0xc9, 0xd4, 255) },
    { "amber",   IM_COL32(0xff, 0xb8, 0x5a, 255) },
    { "magenta", IM_COL32(0xc8, 0x78, 0xd6, 255) },
    { "lime",    IM_COL32(0x9c, 0xd0, 0x52, 255) },
    { "rose",    IM_COL32(0xe0, 0x52, 0x4a, 255) },
};

}  // namespace

std::filesystem::path SettingsPath() {
    if (const char* x = std::getenv("XDG_CONFIG_HOME"); x && *x) {
        return std::filesystem::path(x) / "llobotomy" / "settings.json";
    }
    if (const char* h = std::getenv("HOME"); h && *h) {
        return std::filesystem::path(h) / ".config" / "llobotomy" / "settings.json";
    }
    return std::filesystem::path("./settings.json");
}

void SettingsLoad(AppState& s) {
    const auto p = SettingsPath();
    std::ifstream in(p);
    if (!in) {
        LLOB_LOG_DEBUG("settings", "no settings file at %s — using defaults", p.c_str());
        ReapplyStyle(s);
        return;
    }

    // Minimal scanner: tokenize on `:` and `,` boundaries, treat anything
    // outside the schema as a no-op.  Matches what SettingsSave writes.
    std::stringstream buf; buf << in.rdbuf();
    const std::string body = buf.str();

    std::size_t pos = 0;
    while (pos < body.size()) {
        const auto colon = body.find(':', pos);
        if (colon == std::string::npos) break;
        const auto key_end = body.rfind('"', colon);
        if (key_end == std::string::npos) break;
        const auto key_start = body.rfind('"', key_end - 1);
        if (key_start == std::string::npos) break;
        const std::string_view key(body.data() + key_start + 1, key_end - key_start - 1);

        const auto val_end = body.find_first_of(",}\n", colon + 1);
        if (val_end == std::string::npos) break;
        const std::string_view raw(body.data() + colon + 1, val_end - colon - 1);
        const std::string_view val = trim(raw);
        pos = val_end + 1;

        if      (key == "theme")    { Theme t;    if (ThemeFromName(val, t))   s.theme   = t; }
        else if (key == "density")  { Density d;  if (DensityFromName(val, d)) s.density = d; }
        else if (key == "accent") {
            if (val == "default") { s.accentOverride = 0; }
            else {
                const std::string v(val);
                char* end = nullptr;
                const auto x = std::strtoul(v.c_str(), &end, 0);  // 0x... or decimal
                if (end != v.c_str()) s.accentOverride = static_cast<ImU32>(x) | 0xff000000u;
            }
        }
        else if (key == "showRaw")  { s.showRaw  = (val == "true" || val == "1"); }
        else if (key == "liveAnim") { s.liveAnim = (val == "true" || val == "1"); }
    }

    LLOB_LOG_INFO("settings", "loaded %s (theme=%s density=%s accent=%s)",
                  p.c_str(), ThemeName(s.theme), DensityName(s.density),
                  s.accentOverride ? "custom" : "default");
    ReapplyStyle(s);
}

bool SettingsSave(const AppState& s) {
    const auto p = SettingsPath();
    EnsureParentDir(p);

    const auto tmp = p.string() + ".tmp";
    std::ofstream out(tmp);
    if (!out) {
        LLOB_LOG_WARN("settings", "could not open %s for write", tmp.c_str());
        return false;
    }

    char accent_buf[24];
    if (s.accentOverride == 0) {
        std::snprintf(accent_buf, sizeof accent_buf, "\"default\"");
    } else {
        std::snprintf(accent_buf, sizeof accent_buf, "\"0x%06x\"",
                      static_cast<unsigned>(s.accentOverride & 0x00ffffffu));
    }

    out << "{\n"
        << "  \"theme\": \""    << ThemeName(s.theme)     << "\",\n"
        << "  \"density\": \""  << DensityName(s.density) << "\",\n"
        << "  \"accent\": "     << accent_buf             << ",\n"
        << "  \"showRaw\": "    << (s.showRaw  ? "true" : "false") << ",\n"
        << "  \"liveAnim\": "   << (s.liveAnim ? "true" : "false") << "\n"
        << "}\n";
    out.close();

    std::error_code ec;
    std::filesystem::rename(tmp, p, ec);
    if (ec) {
        LLOB_LOG_WARN("settings", "rename %s -> %s failed: %s",
                      tmp.c_str(), p.c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

void DrawSettingsModal(AppState& s, bool request_open) {
    if (request_open) ImGui::OpenPopup("settings##modal");

    // Centered, fixed-width modal — feels more like a Photoshop preferences
    // dialog than a free-floating window.
    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(440, 0), ImGuiCond_Appearing);

    if (!ImGui::BeginPopupModal("settings##modal", nullptr,
                                ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse)) {
        return;
    }

    bool dirty = false;

    // ── Theme ─────────────────────────────────────────────────────────────
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
    ImGui::TextUnformatted("THEME");
    ImGui::PopStyleColor();
    {
        int t = static_cast<int>(s.theme);
        if (ImGui::RadioButton("tracy##theme",     &t, 0)) { s.theme = Theme::Tracy;     dirty = true; }
        ImGui::SameLine();
        if (ImGui::RadioButton("photoshop##theme", &t, 1)) { s.theme = Theme::Photoshop; dirty = true; }
        ImGui::SameLine();
        if (ImGui::RadioButton("amber##theme",     &t, 2)) { s.theme = Theme::Amber;     dirty = true; }
    }

    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    // ── Density ───────────────────────────────────────────────────────────
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
    ImGui::TextUnformatted("DENSITY");
    ImGui::PopStyleColor();
    {
        int d = static_cast<int>(s.density);
        if (ImGui::RadioButton("compact##den",  &d, 0)) { s.density = Density::Compact;  dirty = true; }
        ImGui::SameLine();
        if (ImGui::RadioButton("normal##den",   &d, 1)) { s.density = Density::Normal;   dirty = true; }
        ImGui::SameLine();
        if (ImGui::RadioButton("spacious##den", &d, 2)) { s.density = Density::Spacious; dirty = true; }
    }

    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    // ── Accent override ───────────────────────────────────────────────────
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
    ImGui::TextUnformatted("ACCENT OVERRIDE");
    ImGui::PopStyleColor();
    {
        bool override_on = (s.accentOverride != 0);
        if (ImGui::Checkbox("override theme accent", &override_on)) {
            s.accentOverride = override_on
                ? (kAccentPresets[0].color | 0xff000000u)
                : 0u;
            dirty = true;
        }

        ImGui::BeginDisabled(!override_on);
        for (size_t i = 0; i < std::size(kAccentPresets); ++i) {
            ImGui::PushID(static_cast<int>(i));
            const ImVec4 col = ImGui::ColorConvertU32ToFloat4(kAccentPresets[i].color);
            if (ImGui::ColorButton("##swatch", col, ImGuiColorEditFlags_NoTooltip, ImVec2(28, 28))) {
                s.accentOverride = kAccentPresets[i].color;
                dirty = true;
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kAccentPresets[i].name);
            ImGui::PopID();
            if (i + 1 < std::size(kAccentPresets)) ImGui::SameLine();
        }

        ImVec4 custom = override_on
            ? ImGui::ColorConvertU32ToFloat4(s.accentOverride)
            : ImVec4(0, 0, 0, 1);
        if (ImGui::ColorEdit3("custom##accent", &custom.x, ImGuiColorEditFlags_NoInputs)) {
            s.accentOverride = ImGui::ColorConvertFloat4ToU32(custom) | 0xff000000u;
            dirty = true;
        }
        ImGui::EndDisabled();
    }

    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    // ── Toggles ───────────────────────────────────────────────────────────
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_muted);
    ImGui::TextUnformatted("PANELS");
    ImGui::PopStyleColor();
    if (ImGui::Checkbox("show raw pane (architecture workspace)", &s.showRaw))  dirty = true;
    if (ImGui::Checkbox("animate live data (4Hz tick)",            &s.liveAnim)) dirty = true;

    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    // ── Footer ────────────────────────────────────────────────────────────
    ImGui::PushStyleColor(ImGuiCol_Text, Sty().text_dim);
    ImGui::Text("config: %s", SettingsPath().c_str());
    ImGui::PopStyleColor();
    ImGui::Spacing();
    if (ImGui::Button("close", ImVec2(120, 0))) ImGui::CloseCurrentPopup();

    if (dirty) {
        ReapplyStyle(s);
        SettingsSave(s);
    }

    ImGui::EndPopup();
}

}  // namespace llob
